"""
run_reasoning.py — the end-to-end inspection run.

Loads the knowledge base, the scene, the observations and the observation conditions; generates
inspection items from requirement applicability; assesses evidence sufficiency; decides each item;
and writes the decisions back into the graph with their full trace.

On the two implementations of the rules
---------------------------------------
rules/inspection_rules.rq holds the canonical SPARQL CONSTRUCT rules, and those are what a Jena
Fuseki or GraphDB deployment executes. This module reimplements the same item generation and
evidence marshalling in Python against the dependency-free parser in ttl_tools, so that the whole
chain can be run and checked in an environment with no triple store. The two must agree; when they
diverge, the SPARQL file is authoritative and this module is the bug.

The numeric sufficiency computation is not duplicated: it lives only in sufficiency.py, because
expressing a calibrated logistic and a Bayesian update in SPARQL would be neither readable nor
checkable.

Run:
    python3 src/run_reasoning.py --model bayesian
    python3 src/run_reasoning.py --model naive      # the closed-world ablation arm
    python3 src/run_reasoning.py --ablation         # all three arms, compared
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ttl_tools import IRI, Graph, Literal, parse_file, merge  # noqa: E402
from csv_to_rdf import slug  # noqa: E402  — the single source of truth for IRI local names
from detectability import (  # noqa: E402
    DetectabilityModel, ObservationCondition, PrecisionCalibration, load_object_classes,
)
from sufficiency import (  # noqa: E402
    SufficiencyAssessor, SufficiencyModelKind, SufficiencyThresholds,
)
from decision import (  # noqa: E402
    ComplianceState, Detection, EvidencePattern, InspectionRuleEngine, Requirement,
)

INSP = "https://w3id.org/cieo/inspection#"
CST = "https://w3id.org/cieo/construction#"
OBS = "https://w3id.org/cieo/observation#"
EVD = "https://w3id.org/cieo/evidence#"
REG = "https://w3id.org/cieo/regulation#"
CIEO = "https://w3id.org/cieo/core#"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


class Index:
    """A minimal SPO index over a parsed graph. Enough for the joins the rules need."""

    def __init__(self, g: Graph):
        self.spo: dict = defaultdict(lambda: defaultdict(list))
        self.pos: dict = defaultdict(lambda: defaultdict(list))
        for s, p, o in g.triples:
            if not isinstance(p, IRI):
                continue
            self.spo[s][p.value].append(o)
            key = o.value if isinstance(o, IRI) else (o.value if isinstance(o, Literal) else str(o))
            self.pos[p.value][key].append(s)

    def one(self, s, p: str):
        v = self.spo.get(s, {}).get(p, [])
        return v[0] if v else None

    def all(self, s, p: str) -> list:
        return self.spo.get(s, {}).get(p, [])

    def subjects_of_type(self, cls: str) -> list:
        return self.pos.get(RDF_TYPE, {}).get(cls, [])

    def lit(self, s, p: str) -> str | None:
        v = self.one(s, p)
        return v.value if isinstance(v, Literal) else None

    def num(self, s, p: str) -> float | None:
        v = self.lit(s, p)
        try:
            return float(v) if v is not None else None
        except ValueError:
            return None


def local(term) -> str:
    if isinstance(term, IRI):
        return term.value.rsplit("#", 1)[-1]
    return str(term)


def load_graph(paths: list[Path]) -> tuple[Graph, Index]:
    graphs = []
    for p in paths:
        if p.exists():
            graphs.append(parse_file(p))
        else:
            print(f"  (skipping missing {p})")
    g = merge(graphs)
    return g, Index(g)


# ---------------------------------------------------------------------------
# R00 — requirement applicability
# ---------------------------------------------------------------------------

def generate_items(idx: Index) -> list[dict]:
    """
    One item per (requirement, target) where the requirement is scoped to a phase the target is
    live in and the target is of the type the requirement addresses.

    Requirements that generate no items are not silent failures: they are obligations that do not
    currently bind, and the summary counts them separately so the difference stays visible.
    """
    items = []
    for req in idx.subjects_of_type(INSP + "InspectionRequirement"):
        req_id = idx.lit(req, INSP + "requirementId") or local(req)
        phases = {local(p) for p in idx.all(req, INSP + "appliesToPhase")}
        target_types = idx.all(req, INSP + "appliesToTargetType")

        for tt in target_types:
            if not isinstance(tt, IRI):
                continue
            for target in idx.subjects_of_type(tt.value):
                live = {local(p) for p in idx.all(target, CST + "existsDuringPhase")}
                live |= {local(p) for p in idx.all(target, CST + "belongsToPhase")}
                hit = phases & live
                if not hit:
                    continue
                items.append({
                    "item_id": f"ITEM_{req_id}_{local(target)}",
                    "req_node": req,
                    "req_id": req_id,
                    "target": target,
                    "target_id": local(target),
                    "phase": sorted(hit)[0],
                })
    return items


def build_requirement(idx: Index, node, req_id: str) -> Requirement:
    pattern_iri = idx.one(node, INSP + "hasEvidencePattern")
    pattern_name = local(pattern_iri) if pattern_iri else "PresenceOfRequired"
    try:
        pattern = EvidencePattern(pattern_name)
    except ValueError:
        pattern = EvidencePattern.PRESENCE_OF_REQUIRED

    def labels(prop: str) -> list[str]:
        out = []
        for c in idx.all(node, prop):
            lbl = idx.lit(c, OBS + "detectorLabel")
            if lbl:
                out.append(lbl)
        return out

    clause = idx.one(node, INSP + "specifiedBy")
    clause_ref = idx.lit(clause, REG + "clauseReference") if clause else ""
    regulation = ""
    if clause is not None:
        for src in idx.pos.get(REG + "hasClause", {}).get(
                clause.value if isinstance(clause, IRI) else "", []):
            regulation = local(src)
            break

    checkability = idx.one(node, INSP + "hasCheckability")
    return Requirement(
        requirement_id=req_id,
        pattern=pattern,
        required_classes=labels(INSP + "requiresObjectClass"),
        prohibited_classes=labels(INSP + "prohibitsObjectClass"),
        required_coverage=idx.num(node, INSP + "requiredCoverage"),
        clause_ref=clause_ref or "",
        regulation_id=regulation,
        checkability=(local(checkability).upper() if checkability else "MANUAL"),
        statement=idx.lit(node, INSP + "requirementStatement") or "",
    )


# ---------------------------------------------------------------------------
# Evidence marshalling
# ---------------------------------------------------------------------------

def conditions_for(idx: Index, target) -> list[ObservationCondition]:
    out = []
    for cond in idx.subjects_of_type(EVD + "ObservationCondition"):
        if idx.one(cond, EVD + "assessesTarget") != target:
            continue
        vp = idx.one(cond, EVD + "fromViewpoint")
        img = idx.one(cond, EVD + "conditionOfImage")
        mask = idx.lit(cond, CIEO + "hasIdentifier") or ""
        out.append(ObservationCondition(
            target_id=local(target),
            viewpoint_id=local(vp) if vp else "",
            image_id=local(img) if img else "",
            target_extent=idx.num(target, CIEO + "hasSpatialExtent") or 0.0,
            extent_unit=idx.lit(target, CIEO + "extentUnit") or "m",
            frustum_coverage=idx.num(cond, EVD + "frustumCoverage") or 0.0,
            unoccluded_fraction=idx.num(cond, EVD + "unoccludedFraction") or 0.0,
            effective_coverage=idx.num(cond, EVD + "effectiveCoverage") or 0.0,
            viewing_distance_m=idx.num(cond, EVD + "viewingDistance") or 0.0,
            incidence_angle_deg=idx.num(cond, EVD + "incidenceAngle") or 0.0,
            gsd_along_surface_m=idx.num(cond, EVD + "groundSampleDistance") or 0.0,
            sampled_point_count=int(idx.num(cond, EVD + "sampledPointCount") or 0),
            condition_valid=(idx.lit(cond, EVD + "conditionValid") != "false"),
            visible_sample_mask=mask[5:] if mask.startswith("mask:") else "",
        ))
    return out


def detections_for(idx: Index, target) -> list[Detection]:
    """Positive detections grounded to this target. Non-detections need no lifting here: the
    decision engine treats the absence of a positive as the absence, and the graph's explicit
    NonDetection nodes are what make that absence an assertion rather than a gap."""
    out = []
    for g in idx.subjects_of_type(OBS + "SpatialGrounding"):
        if idx.one(g, OBS + "groundedTo") != target:
            continue
        obs_node = idx.one(g, OBS + "groundsObservation")
        if obs_node is None:
            continue
        types = {local(t) for t in idx.all(obs_node, RDF_TYPE)}
        if "VisualDetection" not in types:
            continue
        cls = idx.one(obs_node, OBS + "detectedClass")
        label = idx.lit(cls, OBS + "detectorLabel") if cls else None
        if not label:
            continue
        img = idx.one(obs_node, OBS + "fromImage")
        out.append(Detection(
            detection_id=local(obs_node),
            detector_label=label,
            confidence=idx.num(obs_node, OBS + "detectionConfidence") or 0.0,
            image_id=local(img) if img else "",
            grounding_confidence=idx.num(g, OBS + "groundingConfidence") or 1.0,
            grounding_ambiguous=(idx.lit(g, OBS + "groundingAmbiguous") == "true"),
        ))
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def decisions_to_ttl(decisions: list, model_kind: SufficiencyModelKind) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "@prefix sav:  <https://w3id.org/cieo/savigliano#> .",
        "@prefix kb:   <https://w3id.org/cieo/kb#> .",
        "@prefix insp: <https://w3id.org/cieo/inspection#> .",
        "@prefix evd:  <https://w3id.org/cieo/evidence#> .",
        "@prefix cmp:  <https://w3id.org/cieo/compliance#> .",
        "@prefix prov: <http://www.w3.org/ns/prov#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
        "",
        f"# Generated by src/run_reasoning.py, sufficiency model {model_kind.value}, {now}.",
        "",
    ]

    def esc(t: str) -> str:
        return str(t).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")

    for d in decisions:
        node = f"sav:{d.item_id}_DEC"
        suff_node = f"sav:{d.item_id}_SUFF"
        lines.append(f"sav:{d.item_id} a insp:InspectionItem .")

        if d.sufficiency is not None:
            s = d.sufficiency
            lines.append(f"{suff_node} a evd:EvidenceSufficiency ;")
            lines.append(f"    evd:assessesItem sav:{d.item_id} ;")
            lines.append(f"    evd:usesModel evd:{s.model.value} ;")
            lines.append(f"    evd:hasSufficiencyLevel evd:{s.level.value} ;")
            lines.append(f"    evd:aggregatedRecall {s.aggregated_recall:.6f} ;")
            lines.append(f"    evd:priorPresence {s.prior_presence:.4f} ;")
            lines.append(f"    evd:posteriorAbsence {s.posterior_absence:.6f} ;")
            lines.append(f"    evd:coverageAchieved {s.coverage_achieved:.4f} ;")
            lines.append(f"    evd:temporalValidity {s.temporal_validity:.4f} ;")
            lines.append(f'    evd:sufficiencyRationale "{esc(s.rationale)}" .')
            lines.append("")

        lines.append(f"{node} a cmp:InspectionDecision ;")
        lines.append(f"    cmp:decidesItem sav:{d.item_id} ;")
        lines.append(f"    cmp:hasState cmp:{d.state.value} ;")
        lines.append(f"    cmp:appliedRule cmp:{d.rule_id} ;")
        if d.sufficiency is not None:
            lines.append(f"    cmp:gatedBy {suff_node} ;")
        if d.cites_clause:
            # Must use exactly the same slug rule as csv_to_rdf.py, or the citation will point at
            # an IRI no clause occupies and the trace will break silently at its most important
            # link. Importing the function rather than re-implementing it is what keeps them tied.
            lines.append(f"    cmp:citesClause "
                         f"kb:Clause_{d.cites_regulation}_{slug(d.cites_clause)[:40]} ;")
        lines.append(f'    cmp:hasRationale "{esc(d.rationale)}" ;')
        if d.recommended_action:
            lines.append(f'    cmp:recommendedAction "{esc(d.recommended_action)}" ;')
        lines.append(f"    cmp:requiresHumanVerification "
                     f"{'true' if d.requires_human_verification else 'false'} ;")
        lines.append(f'    cmp:decidedAt "{now}"^^xsd:dateTime .')
        lines.append("")

        lines.append(f"sav:{d.item_id}_TRACE a cmp:TraceabilityRecord ;")
        lines.append(f"    cmp:tracesDecision {node} ;")
        lines.append(f"    cmp:traceComplete {'true' if d.trace_complete else 'false'} ;")
        if not d.trace_complete:
            lines.append(f'    cmp:traceBrokenAt "{esc(d.trace_broken_at)}" ;')
        lines.append("    cmp:traceHops 6 .")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------

def run(model_kind: SufficiencyModelKind, calibrated: bool, quiet: bool = False):
    graph_files = [
        Path("data/cieo-kb.ttl"),
        Path("data/savigliano-abox.ttl"),
        Path("data/demo/observations.ttl"),
        Path("data/demo/conditions.ttl"),
    ]
    g, idx = load_graph(graph_files)

    classes = load_object_classes("data/object_classes.csv")
    if calibrated:
        model = DetectabilityModel(
            b_resolution=0.92, b_occlusion=-3.85, b_incidence=-2.10,
            calibrated=True, n_calibration=1240,
            slope_ci={"b_resolution": (0.78, 1.06), "b_occlusion": (-4.40, -3.30),
                      "b_incidence": (-2.60, -1.60)},
            px_range=(4.0, 900.0), incidence_range=(0.0, 85.0),
        )
    else:
        model = DetectabilityModel()

    precision = PrecisionCalibration()
    assessor = SufficiencyAssessor(model, SufficiencyThresholds(), precision)
    eng = InspectionRuleEngine(assessor, classes, precision)

    items = generate_items(idx)
    decisions = []
    context: dict[str, dict] = {}
    for it in items:
        req = build_requirement(idx, it["req_node"], it["req_id"])
        conds = conditions_for(idx, it["target"])
        dets = detections_for(idx, it["target"])
        context[it["item_id"]] = {
            "requirement": req,
            "viewpoints": [c.viewpoint_id for c in conds],
            "images": sorted({c.image_id for c in conds if c.image_id}),
            "phase": it["phase"],
        }
        decisions.append(eng.decide(
            it["item_id"], it["target_id"], req, conds, dets, model_kind=model_kind,
        ))

    if not quiet:
        print(f"\nGraph: {len(g)} triples from {len([p for p in graph_files if p.exists()])} files")
        print(f"Items generated: {len(items)}")
        print(f"Detectability model: {'calibrated' if calibrated else 'UNCALIBRATED'}")
        print(f"Sufficiency model: {model_kind.value}\n")

        counts = defaultdict(int)
        for d in decisions:
            counts[d.state.value] += 1
        width = max((len(k) for k in counts), default=10)
        for state, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {state:<{width}}  {n:3d}")

        print("\n  item                                              state                       "
              "recall  coverage")
        print("  " + "-" * 100)
        for d in sorted(decisions, key=lambda x: x.item_id):
            s = d.sufficiency
            r = f"{s.aggregated_recall:6.3f}" if s else "     -"
            c = f"{s.coverage_achieved:6.2f}" if s else "     -"
            print(f"  {d.item_id[:48]:<48}  {d.state.value:<24}  {r}  {c}")

    return decisions, items, context


def decisions_to_json(decisions, context, model_kind) -> str:
    """
    The same decisions, in the shape unity/InspectionResultPanel.cs reads.

    A flat JSON file rather than a SPARQL endpoint is deliberate for the first build: Unity can
    read it with no server, no Fuseki, no network, and the inspector sees the reasoning
    immediately. Swap it for a live endpoint once the reasoning is settled and you want the
    virtual environment to query the graph rather than a snapshot of it.
    """
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sufficiency_model": model_kind.value,
        "decisions": [],
    }
    for d in sorted(decisions, key=lambda x: x.item_id):
        ctx = context.get(d.item_id, {})
        req = ctx.get("requirement")
        s = d.sufficiency
        payload["decisions"].append({
            "item_id": d.item_id,
            "target_id": d.target_id,
            "requirement_id": d.requirement_id,
            "phase": ctx.get("phase", ""),
            "requirement_statement": getattr(req, "statement", "") if req else "",
            "evidence_pattern": (req.pattern.value if req else ""),
            "state": d.state.value,
            "rule_id": d.rule_id,
            "regulation": d.cites_regulation,
            "clause": d.cites_clause,
            "rationale": d.rationale,
            "recommended_action": d.recommended_action,
            "requires_human_verification": d.requires_human_verification,
            "trace_complete": d.trace_complete,
            "aggregated_recall": round(s.aggregated_recall, 4) if s else None,
            "posterior_absence": round(s.posterior_absence, 4) if s else None,
            "coverage_achieved": round(s.coverage_achieved, 4) if s else None,
            "coverage_required": round(s.coverage_required, 4) if s else None,
            "sufficiency_level": s.level.value if s else "",
            "limiting_class": s.limiting_class if s else "",
            "n_viewpoints": s.n_valid_viewpoints if s else 0,
            "viewpoint_ids": ctx.get("viewpoints", []),
            "image_ids": ctx.get("images", []),
            "supporting_detections": d.supporting_detections,
        })
    return json.dumps(payload, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the CIEO inspection reasoning end to end.")
    ap.add_argument("--model", choices=["bayesian", "heuristic", "naive"], default="bayesian")
    ap.add_argument("--uncalibrated", action="store_true",
                    help="Use the uncalibrated detectability model, which will decline to settle.")
    ap.add_argument("--ablation", action="store_true", help="Run all three arms and compare.")
    ap.add_argument("--out", type=Path, default=Path("evaluation/results/decisions.ttl"))
    args = ap.parse_args()

    kinds = {
        "bayesian": SufficiencyModelKind.BAYESIAN_ABSENCE,
        "heuristic": SufficiencyModelKind.HEURISTIC_SCORE,
        "naive": SufficiencyModelKind.NAIVE_CLOSED_WORLD,
    }

    if args.ablation:
        print("\nAblation: the same evidence under three sufficiency models")
        print("=" * 104)
        results = {}
        for name, kind in kinds.items():
            decisions, _, _ = run(kind, calibrated=True, quiet=True)
            results[name] = {d.item_id: d.state for d in decisions}

        ids = sorted(results["bayesian"])
        print(f"\n  {'item':<46}  {'naive':<22}  {'heuristic':<24}  {'bayesian'}")
        print("  " + "-" * 100)
        for i in ids:
            print(f"  {i[:46]:<46}  {results['naive'][i].value:<22}  "
                  f"{results['heuristic'][i].value:<24}  {results['bayesian'][i].value}")

        settled = {cmp: sum(1 for i in ids if results[cmp][i] in
                            (ComplianceState.COMPLIANT, ComplianceState.NON_COMPLIANT))
                   for cmp in results}
        accusations = {cmp: sum(1 for i in ids if results[cmp][i] == ComplianceState.NON_COMPLIANT)
                       for cmp in results}
        print(f"\n  items                       {len(ids)}")
        for name in ("naive", "heuristic", "bayesian"):
            print(f"  {name:<12} settled {settled[name]:3d}   non-compliance asserted "
                  f"{accusations[name]:3d}")
        avoided = accusations["naive"] - accusations["bayesian"]
        print(f"\n  Non-compliance findings the closed-world arm asserts and the calibrated arm "
              f"declines: {avoided}")
        print("  Each of those is a finding the system would have reported without having "
              "established\n  that the evidence could support it. Whether each is a genuine false "
              "alarm is what the\n  ground-truth comparison in Experiment 2 has to settle.")
        return

    decisions, _, context = run(kinds[args.model], calibrated=not args.uncalibrated)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(decisions_to_ttl(decisions, kinds[args.model]), encoding="utf-8")

    json_out = args.out.with_suffix(".json")
    json_out.write_text(decisions_to_json(decisions, context, kinds[args.model]), encoding="utf-8")

    print(f"\nWrote {args.out}")
    print(f"Wrote {json_out}   <- copy this into your Unity project's StreamingAssets folder")


if __name__ == "__main__":
    main()

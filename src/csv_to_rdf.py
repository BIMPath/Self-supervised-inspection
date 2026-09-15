"""
csv_to_rdf.py — lift the requirements database into the knowledge graph.

The CSVs under data/ are the authoritative source. They are the artefact a domain expert can
review, argue with and correct without touching Turtle, and keeping them authoritative rather than
hand-maintaining the A-Box is what stops the ontology and the checklist drifting apart. This script
regenerates the graph from them; the generated file is never edited by hand.

Run:
    python3 src/csv_to_rdf.py --data data --out data/cieo-kb.ttl
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import date
from pathlib import Path

KB = "https://w3id.org/cieo/kb#"

PREFIXES = """@prefix kb:      <https://w3id.org/cieo/kb#> .
@prefix cieo:    <https://w3id.org/cieo/core#> .
@prefix cst:     <https://w3id.org/cieo/construction#> .
@prefix insp:    <https://w3id.org/cieo/inspection#> .
@prefix reg:     <https://w3id.org/cieo/regulation#> .
@prefix obs:     <https://w3id.org/cieo/observation#> .
@prefix evd:     <https://w3id.org/cieo/evidence#> .
@prefix cmp:     <https://w3id.org/cieo/compliance#> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .
@prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .
@prefix skos:    <http://www.w3.org/2004/02/skos/core#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix prov:    <http://www.w3.org/ns/prov#> .
"""

PATTERN_MAP = {
    "PresenceOfRequired": "insp:PresenceOfRequired",
    "SpatialCompleteness": "insp:SpatialCompleteness",
    "ComponentIntegrity": "insp:ComponentIntegrity",
    "VerifiedAbsence": "insp:VerifiedAbsence",
    "GeometricConformance": "insp:GeometricConformance",
    "DocumentaryConformance": "insp:DocumentaryConformance",
    "MaterialTestConformance": "insp:MaterialTestConformance",
}

CHECKABILITY_MAP = {
    "AUTOMATIC": "insp:Automatic",
    "ASSISTED": "insp:Assisted",
    "MANUAL": "insp:Manual",
}

MODALITY_MAP = {
    "Visual": "insp:VisualModality",
    "Geometric": "insp:GeometricModality",
    "Documentary": "insp:DocumentaryModality",
    "MaterialTest": "insp:MaterialTestModality",
}

CONFIDENCE_MAP = {
    "HIGH": "reg:VerifiedClause",
    "MEDIUM": "reg:ProbableClause",
    "TO_VERIFY": "reg:UnverifiedClause",
}

INSTRUMENT_MAP = {
    "LegislativeDecree": "reg:LegislativeDecree",
    "MinisterialDecree": "reg:MinisterialDecree",
    "Circular": "reg:Circular",
    "HarmonisedStandard": "reg:HarmonisedStandard",
    "NationalStandard": "reg:NationalStandard",
    "ProductStandard": "reg:ProductStandard",
    "ApprovalGuideline": "reg:ApprovalGuideline",
    "ContractSpecification": "reg:ContractSpecification",
}

# Concealing activities, and which hold-point requirements they close off. Named explicitly
# rather than derived, because which activity conceals what is a construction-sequencing fact
# and getting it wrong means a hold point that never fires.
CONCEALING_ACTIVITIES = {
    "AbutmentConcretePour": {
        "label": "Abutment concrete pour",
        "phase": "P3",
        "conceals": ["IR-P3-009", "IR-P3-010", "IR-P3-015"],
    },
    "DeckSlabConcretePour": {
        "label": "Deck slab concrete pour",
        "phase": "P5",
        "conceals": ["IR-P5-026"],
    },
    "UtilityBackfill": {
        "label": "Utility trench backfill",
        "phase": "P5",
        "conceals": ["IR-P5-028"],
    },
    "WearingCourseLaying": {
        "label": "Wearing course laying",
        "phase": "P6",
        "conceals": ["IR-P6-030"],
    },
    "SteelErection": {
        "label": "Steel superstructure erection",
        "phase": "P4",
        "conceals": ["IR-P4-017", "IR-P4-018", "IR-P4-020", "IR-P4-021", "IR-P4-023"],
    },
    "ConcreteBatchPlacement": {
        "label": "Concrete batch placement",
        "phase": "P3",
        "conceals": ["IR-P3-011", "IR-P3-014"],
    },
}

DEFAULT_SPATIAL_COVERAGE = 0.95


def slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", text.strip())
    return re.sub(r"_+", "_", s).strip("_")


def esc(text: str) -> str:
    """Escape a string for a Turtle literal."""
    return (text.replace("\\", "\\\\").replace('"', '\\"')
                .replace("\n", "\\n").replace("\r", "").replace("\t", "\\t"))


def lit(text: str, lang: str | None = None) -> str:
    return f'"{esc(text)}"' + (f"@{lang}" if lang else "")


def split_list(cell: str) -> list[str]:
    return [x.strip() for x in (cell or "").split(";") if x.strip()]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------

def emit_phases(rows: list[dict]) -> list[str]:
    out = ["\n" + "#" * 78, "# Construction phases", "#" * 78 + "\n"]
    for r in rows:
        pid = r["phase_id"]
        out.append(f"kb:{pid} a cst:ConstructionPhase ;")
        out.append(f'    rdfs:label {lit(r["phase_label"], "en")} ;')
        out.append(f'    rdfs:comment {lit(r["description"], "en")} ;')
        out.append(f'    cst:phaseSequence {r["sequence"]} ;')
        for succ in split_list(r.get("precedes", "")):
            out.append(f"    cst:phasePrecedes kb:{succ} ;")
        out[-1] = out[-1].rstrip(" ;") + " ."
        out.append("")
    return out


def emit_regulations(rows: list[dict]) -> list[str]:
    out = ["\n" + "#" * 78, "# Regulatory sources", "#" * 78 + "\n"]
    for r in rows:
        rid = slug(r["regulation_id"])
        cls = INSTRUMENT_MAP.get(r["instrument_type"], "reg:RegulatorySource")
        out.append(f"kb:{rid} a {cls} ;")
        out.append(f'    reg:shortName {lit(r["short_name"])} ;')
        out.append(f'    reg:fullTitle {lit(r["full_title"])} ;')
        out.append(f'    reg:inJurisdiction reg:{r["jurisdiction"]} ;')
        out.append(f'    rdfs:label {lit(r["short_name"], "en")} .')
        out.append("")
    return out


def emit_object_classes(rows: list[dict]) -> list[str]:
    out = ["\n" + "#" * 78,
           "# Detector object classes, with the physical parameters that decide observability",
           "#" * 78 + "\n"]
    label_to_iri: dict[str, str] = {}
    for r in rows:
        cid = f"Class_{slug(r['detector_label'])}"
        label_to_iri[r["detector_label"]] = f"kb:{cid}"
        out.append(f"kb:{cid} a obs:ObjectClass ;")
        out.append(f'    skos:prefLabel {lit(r["semantic_class"], "en")} ;')
        out.append(f'    obs:detectorLabel {lit(r["detector_label"])} ;')
        out.append(f'    obs:charDimInstalled {float(r["char_dim_installed_m"]):.4f} ;')
        out.append(f'    obs:minPixelsOnTarget {r["min_px_on_target"]} ;')
        out.append(f'    obs:occlusionSensitivity {float(r["occlusion_sensitivity"]):.2f} ;')
        out.append(f'    obs:selfOcclusionPrior {float(r["self_occlusion_prior"]):.2f} ;')
        out.append(f'    obs:baseRecallRef {float(r["base_recall_ref"]):.2f} ;')
        if r.get("notes"):
            out.append(f'    rdfs:comment {lit(r["notes"], "en")} ;')
        out[-1] = out[-1].rstrip(" ;") + " ."
        out.append("")

    # Vocabulary defects: duplicate labels in the trained model, recorded rather than quietly fixed.
    for dup, canonical in (("nuts", "nut"), ("bolt_dup", "bolt")):
        if dup in label_to_iri and canonical in label_to_iri:
            out.append(f"{label_to_iri[dup]} obs:mergedInto {label_to_iri[canonical]} .")
    out.append("")
    return out, label_to_iri


def emit_concealing_activities() -> list[str]:
    out = ["\n" + "#" * 78, "# Concealing activities (hold-point closures)", "#" * 78 + "\n"]
    for aid, spec in CONCEALING_ACTIVITIES.items():
        out.append(f"kb:Activity_{aid} a cst:ConcealingActivity ;")
        out.append(f'    rdfs:label {lit(spec["label"], "en")} ;')
        out.append(f'    cst:belongsToPhase kb:{spec["phase"]} .')
        out.append("")
    return out


def emit_requirements(rows: list[dict], class_iris: dict[str, str]) -> tuple[list[str], list[str]]:
    out = ["\n" + "#" * 78, "# Inspection requirements", "#" * 78 + "\n"]
    warnings: list[str] = []
    seen_clauses: set[str] = set()
    clause_block: list[str] = ["\n" + "#" * 78, "# Regulatory clauses", "#" * 78 + "\n"]

    hold_point_of = {
        rid: aid
        for aid, spec in CONCEALING_ACTIVITIES.items()
        for rid in spec["conceals"]
    }

    for r in rows:
        rid = r["requirement_id"]
        node = f"kb:{slug(rid)}"
        reg_id = slug(r["regulation_id"])
        clause_id = f"Clause_{reg_id}_{slug(r['regulation_clause'])[:40]}"

        if clause_id not in seen_clauses:
            seen_clauses.add(clause_id)
            conf = CONFIDENCE_MAP.get(r.get("clause_confidence", "TO_VERIFY"), "reg:UnverifiedClause")
            clause_block.append(f"kb:{clause_id} a reg:RegulatoryClause ;")
            clause_block.append(f'    reg:clauseReference {lit(r["regulation_clause"])} ;')
            clause_block.append(f"    reg:hasClauseConfidence {conf} .")
            clause_block.append(f"kb:{reg_id} reg:hasClause kb:{clause_id} .")
            clause_block.append("")

        pattern = PATTERN_MAP.get(r["evidence_pattern"])
        if pattern is None:
            warnings.append(f"{rid}: unknown evidence pattern {r['evidence_pattern']!r}")
            continue

        out.append(f"{node} a insp:InspectionRequirement ;")
        out.append(f'    insp:requirementId {lit(rid)} ;')
        out.append(f'    rdfs:label {lit(rid + " " + r["inspection_item"], "en")} ;')
        out.append(f'    insp:inspectionItemLabel {lit(r["inspection_item"])} ;')
        out.append(f'    insp:requirementStatement {lit(r["requirement_statement"])} ;')
        out.append(f'    insp:expectedCondition {lit(r["expected_condition"])} ;')
        out.append(f"    insp:hasEvidencePattern {pattern} ;")
        out.append(f'    insp:appliesToPhase kb:{r["phase_id"]} ;')
        out.append(f'    insp:appliesToTargetType cst:{r["target_type"]} ;')
        out.append(f"    insp:specifiedBy kb:{clause_id} ;")

        chk = CHECKABILITY_MAP.get(r["checkability"])
        if chk:
            out.append(f"    insp:hasCheckability {chk} ;")
        else:
            warnings.append(f"{rid}: unknown checkability {r['checkability']!r}")

        mod = MODALITY_MAP.get(r["evidence_modality"])
        if mod:
            out.append(f"    insp:requiresModality {mod} ;")

        for label in split_list(r.get("required_classes", "")):
            iri = class_iris.get(label)
            if iri:
                out.append(f"    insp:requiresObjectClass {iri} ;")
            else:
                warnings.append(f"{rid}: required class {label!r} is not in the detector vocabulary")

        for label in split_list(r.get("prohibited_classes", "")):
            iri = class_iris.get(label)
            if iri:
                out.append(f"    insp:prohibitsObjectClass {iri} ;")
            else:
                warnings.append(f"{rid}: prohibited class {label!r} is not in the detector vocabulary")

        if r["evidence_pattern"] == "SpatialCompleteness":
            out.append(f"    insp:requiredCoverage {DEFAULT_SPATIAL_COVERAGE} ;")

        if rid in hold_point_of:
            out.append(f"    insp:isHoldPointFor kb:Activity_{hold_point_of[rid]} ;")
        elif r.get("hold_point", "").strip().upper() == "TRUE":
            warnings.append(f"{rid}: marked as a hold point but no concealing activity is mapped to it")

        if r.get("notes"):
            out.append(f'    rdfs:comment {lit(r["notes"], "en")} ;')

        out[-1] = out[-1].rstrip(" ;") + " ."
        out.append("")

    return clause_block + out, warnings


# ---------------------------------------------------------------------------

def build(data_dir: Path) -> tuple[str, list[str]]:
    phases = read_csv(data_dir / "phases.csv")
    regulations = read_csv(data_dir / "regulations.csv")
    classes = read_csv(data_dir / "object_classes.csv")
    requirements = read_csv(data_dir / "inspection_requirements.csv")

    class_block, class_iris = emit_object_classes(classes)
    req_block, warnings = emit_requirements(requirements, class_iris)

    header = [
        PREFIXES,
        "",
        "<https://w3id.org/cieo/kb>",
        "    a owl:Ontology ;",
        f'    dcterms:title {lit("CIEO knowledge base — Savigliano bridge inspection requirements", "en")} ;',
        '    dcterms:description """Generated from the CSVs under data/ by src/csv_to_rdf.py.',
        "",
        "DO NOT EDIT BY HAND. The CSVs are authoritative; edit those and regenerate. Hand-editing",
        "this file is how the checklist a domain expert reviews and the graph the reasoner runs on",
        'stop being the same thing."""@en ;',
        f'    dcterms:created "{date.today().isoformat()}"^^xsd:date ;',
        "    owl:imports <https://w3id.org/cieo> .",
        "",
    ]

    body = (header + emit_phases(phases) + emit_regulations(regulations)
            + class_block + emit_concealing_activities() + req_block)
    return "\n".join(body), warnings


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the CIEO knowledge base from the CSVs.")
    ap.add_argument("--data", default="data", type=Path)
    ap.add_argument("--out", default="data/cieo-kb.ttl", type=Path)
    args = ap.parse_args()

    ttl, warnings = build(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(ttl, encoding="utf-8")

    print(f"Wrote {args.out} ({len(ttl.splitlines())} lines)")
    if warnings:
        print(f"\n{len(warnings)} coverage warning(s) — these are findings to report, not bugs to hide:")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()

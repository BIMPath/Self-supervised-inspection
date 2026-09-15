"""
verify_repo.py — does everything in this repository hold together?

Six checks:

  1. Every Turtle file parses.
  2. Every term in a CIEO namespace that is used anywhere is defined somewhere. This is the check
     that catches the commonest failure in a modular ontology: a property referenced across a
     module boundary that nobody actually declared.
  3. The CSVs and the ontology agree — every target type named in the requirements database exists
     as a class, every object class named exists in the vocabulary.
  4. The generated knowledge base is complete with respect to the CSVs.
  5. The central SHACL guarantee holds in generated output: no decision in a settled state without
     a sufficiency assessment at level Sufficient behind it.
  6. Coverage of the regulatory checklist is reported honestly — how many of the 34 items are
     reachable by the trained detector, and where the gaps are.

Check 5 is the one that matters most. The framework's whole claim is that it will not assert
compliance or non-compliance without having established that the evidence could bear the claim.
That claim is only worth anything if it is verified against real output rather than asserted in
prose, which is what this does.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttl_tools import BNode, Graph, IRI, Literal, parse_file, TurtleSyntaxError  # noqa: E402

CIEO_NS = "https://w3id.org/cieo/"
FAILURES: list[str] = []
WARNINGS: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"    FAIL  {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"    warn  {msg}")


def ok(msg: str) -> None:
    print(f"    ok    {msg}")


# ---------------------------------------------------------------------------

def check_parse() -> dict[str, Graph]:
    print("\n[1] Turtle syntax")
    graphs: dict[str, Graph] = {}
    files = sorted(
        list((ROOT / "ontology").glob("*.ttl"))
        + list((ROOT / "shapes").glob("*.ttl"))
        + list((ROOT / "data").glob("*.ttl"))
        + list((ROOT / "data" / "demo").glob("*.ttl"))
        + list((ROOT / "evaluation" / "results").glob("*.ttl"))
    )
    total = 0
    for f in files:
        try:
            g = parse_file(f)
            graphs[str(f.relative_to(ROOT))] = g
            total += len(g)
        except TurtleSyntaxError as e:
            fail(f"{f.relative_to(ROOT)}: {e}")
    if not FAILURES:
        ok(f"{len(files)} files parsed, {total} triples total")
    return graphs


def check_term_resolution(graphs: dict[str, Graph]) -> None:
    """Every CIEO term used must be declared. Declared means: appears as a subject somewhere."""
    print("\n[2] Cross-module term resolution")

    ontology_files = [k for k in graphs if k.startswith("ontology/")]
    declared: set[str] = set()
    for k in ontology_files:
        for s, _, _ in graphs[k].triples:
            if isinstance(s, IRI) and s.value.startswith(CIEO_NS):
                declared.add(s.value)

    # Terms used in predicate position, or as the object of rdfs:subClassOf / rdfs:range /
    # rdfs:domain / rdf:type, are the ones that must resolve.
    structural = {
        "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "http://www.w3.org/2000/01/rdf-schema#range",
        "http://www.w3.org/2000/01/rdf-schema#domain",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2002/07/owl#onProperty",
    }
    used: dict[str, set[str]] = defaultdict(set)
    for k, g in graphs.items():
        for s, p, o in g.triples:
            if isinstance(p, IRI) and p.value.startswith(CIEO_NS):
                used[p.value].add(k)
            if isinstance(p, IRI) and p.value in structural:
                if isinstance(o, IRI) and o.value.startswith(CIEO_NS):
                    used[o.value].add(k)

    # Terms in the kb#, savigliano# and shapes# namespaces are instance data, not vocabulary.
    instance_ns = (CIEO_NS + "kb#", CIEO_NS + "savigliano#", CIEO_NS + "shapes#")
    undefined = {
        t: srcs for t, srcs in used.items()
        if t not in declared and not t.startswith(instance_ns)
    }

    if undefined:
        for t, srcs in sorted(undefined.items()):
            fail(f"undefined term {t}  (used in {', '.join(sorted(srcs))})")
    else:
        ok(f"{len(used)} distinct CIEO terms used, all resolve to a declaration "
           f"({len(declared)} declared across {len(ontology_files)} modules)")

    unused = declared - set(used) - {
        v for v in declared if v.endswith(("core", "construction", "regulation", "inspection",
                                           "observation", "evidence", "compliance",
                                           "ifc-alignment", "cieo"))
    }
    if unused:
        warn(f"{len(unused)} declared terms are never used in any file "
             f"(fine for a vocabulary, worth a look before publication)")


def check_csv_ontology_agreement(graphs: dict[str, Graph]) -> None:
    print("\n[3] CSV / ontology agreement")

    construction = graphs.get("ontology/cieo-construction.ttl")
    classes = set()
    for s, p, o in construction.triples:
        if isinstance(s, IRI) and s.value.startswith(CIEO_NS + "construction#"):
            classes.add(s.value.rsplit("#", 1)[-1])

    labels = set()
    with open(ROOT / "data" / "object_classes.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels.add(row["detector_label"])

    bad_types, bad_classes, n = [], [], 0
    with open(ROOT / "data" / "inspection_requirements.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            n += 1
            if row["target_type"] not in classes:
                bad_types.append(f"{row['requirement_id']} -> cst:{row['target_type']}")
            for col in ("required_classes", "prohibited_classes"):
                for lbl in (x.strip() for x in (row[col] or "").split(";") if x.strip()):
                    if lbl not in labels:
                        bad_classes.append(f"{row['requirement_id']} -> {lbl!r}")

    for b in bad_types:
        fail(f"target type not defined in the construction module: {b}")
    for b in bad_classes:
        fail(f"object class not in the detector vocabulary: {b}")
    if not bad_types and not bad_classes:
        ok(f"{n} requirements: every target type resolves to a cst: class, every named object "
           f"class is in the vocabulary")


def check_kb_completeness(graphs: dict[str, Graph]) -> None:
    print("\n[4] Generated knowledge base completeness")
    kb = graphs.get("data/cieo-kb.ttl")
    if kb is None:
        fail("data/cieo-kb.ttl is missing; run src/csv_to_rdf.py")
        return

    def count(cls: str) -> int:
        target = IRI(cls)
        return len({s for s, p, o in kb.triples
                    if isinstance(p, IRI) and p.value.endswith("22-rdf-syntax-ns#type")
                    and o == target})

    csv_n = sum(1 for _ in csv.DictReader(
        open(ROOT / "data" / "inspection_requirements.csv", newline="", encoding="utf-8")))
    kb_n = count(CIEO_NS + "inspection#InspectionRequirement")
    if kb_n != csv_n:
        fail(f"{csv_n} requirements in the CSV but {kb_n} in the generated graph; regenerate")
    else:
        ok(f"{kb_n} requirements, {count(CIEO_NS + 'construction#ConstructionPhase')} phases, "
           f"{count(CIEO_NS + 'observation#ObjectClass')} object classes, "
           f"{count(CIEO_NS + 'regulation#RegulatoryClause')} clauses")

    # Every requirement must reach a clause.
    spec = CIEO_NS + "inspection#specifiedBy"
    reqs = {s for s, p, o in kb.triples
            if isinstance(p, IRI) and p.value.endswith("22-rdf-syntax-ns#type")
            and o == IRI(CIEO_NS + "inspection#InspectionRequirement")}
    grounded = {s for s, p, o in kb.triples if isinstance(p, IRI) and p.value == spec}
    missing = reqs - grounded
    if missing:
        fail(f"{len(missing)} requirements have no regulatory clause")
    else:
        ok("every requirement is grounded in a regulatory clause")


def check_settled_decisions_are_gated(graphs: dict[str, Graph]) -> None:
    """The framework's central guarantee, checked against real generated output."""
    print("\n[5] Settled decisions are gated by sufficiency  (the central guarantee)")

    dec = graphs.get("evaluation/results/decisions.ttl")
    if dec is None:
        warn("no decisions file found; run src/run_reasoning.py first")
        return

    CMP = CIEO_NS + "compliance#"
    EVD = CIEO_NS + "evidence#"
    TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

    state_of, gated_by, level_of, cites = {}, {}, {}, {}
    for s, p, o in dec.triples:
        if not isinstance(p, IRI):
            continue
        if p.value == CMP + "hasState":
            state_of[s] = o
        elif p.value == CMP + "gatedBy":
            gated_by[s] = o
        elif p.value == EVD + "hasSufficiencyLevel":
            level_of[s] = o
        elif p.value == CMP + "citesClause":
            cites[s] = o

    settled = {IRI(CMP + "Compliant"), IRI(CMP + "NonCompliant")}
    violations, checked = [], 0
    for d, st in state_of.items():
        if st not in settled:
            continue
        checked += 1
        suff = gated_by.get(d)
        if suff is None:
            violations.append(f"{d} is {st.value.rsplit('#')[-1]} with no gating sufficiency")
        elif level_of.get(suff) != IRI(EVD + "Sufficient"):
            lv = level_of.get(suff)
            violations.append(
                f"{d} is {st.value.rsplit('#')[-1]} but its sufficiency is "
                f"{lv.value.rsplit('#')[-1] if isinstance(lv, IRI) else 'unset'}")

    for v in violations:
        fail(v)
    if not violations:
        ok(f"{len(state_of)} decisions, {checked} in a settled state, every one gated by a "
           f"sufficiency assessment at level Sufficient")

    uncited = [d for d, st in state_of.items()
               if st != IRI(CMP + "NotApplicable") and d not in cites]
    if uncited:
        warn(f"{len(uncited)} decisions cite no regulatory clause")
    else:
        ok("every decision cites a regulatory clause")

    # The citation must land on a clause that exists. A citation pointing at an IRI no clause
    # occupies breaks the trace at its most important link and does so silently, because the
    # decision still looks fully formed.
    kb = graphs.get("data/cieo-kb.ttl")
    if kb is not None:
        clause_iris = {s for s, p, o in kb.triples
                       if isinstance(p, IRI) and p.value == TYPE
                       and o == IRI(CIEO_NS + "regulation#RegulatoryClause")}
        dangling = {c for c in cites.values() if c not in clause_iris}
        if dangling:
            for c in sorted(dangling, key=str)[:5]:
                fail(f"cited clause does not exist in the knowledge base: {c}")
        else:
            ok(f"every cited clause resolves to a clause in the knowledge base "
               f"({len(set(cites.values()))} distinct clauses cited)")


def check_checklist_coverage() -> None:
    print("\n[6] Regulatory checklist coverage  (a finding to report, not a defect to hide)")
    rows = list(csv.DictReader(
        open(ROOT / "data" / "inspection_requirements.csv", newline="", encoding="utf-8")))

    by_check = defaultdict(int)
    by_pattern = defaultdict(int)
    gaps = []
    for r in rows:
        by_check[r["checkability"]] += 1
        by_pattern[r["evidence_pattern"]] += 1
        if "COVERAGE GAP" in (r.get("notes") or ""):
            gaps.append((r["requirement_id"], r["inspection_item"]))

    n = len(rows)
    auto = by_check.get("AUTOMATIC", 0)
    assisted = by_check.get("ASSISTED", 0)
    print(f"    {n} regulatory inspection items from Table 1")
    for k, v in sorted(by_check.items(), key=lambda kv: -kv[1]):
        print(f"      {k:<10} {v:3d}  ({v / n:.0%})")
    print(f"    evidence patterns:")
    for k, v in sorted(by_pattern.items(), key=lambda kv: -kv[1]):
        print(f"      {k:<26} {v:3d}")
    print(f"    detector-vocabulary gaps: {len(gaps)}")
    for rid, item in gaps:
        print(f"      {rid}  {item}")
    ok(f"{auto}/{n} ({auto / n:.0%}) automatically checkable, "
       f"{auto + assisted}/{n} ({(auto + assisted) / n:.0%}) machine-assisted or better")


def check_tests() -> None:
    print("\n[7] Behaviour tests")
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "test_archetypes.py")],
                       capture_output=True, text=True, cwd=ROOT)
    last = [l for l in r.stdout.strip().splitlines() if "passed" in l]
    if r.returncode == 0:
        ok(last[-1] if last else "test suite passed")
    else:
        fail(f"test suite failed: {last[-1] if last else r.stdout[-300:]}")


def main() -> None:
    print("=" * 78)
    print("CIEO repository verification")
    print("=" * 78)

    graphs = check_parse()
    if graphs:
        check_term_resolution(graphs)
        check_csv_ontology_agreement(graphs)
        check_kb_completeness(graphs)
        check_settled_decisions_are_gated(graphs)
    check_checklist_coverage()
    check_tests()

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S), {len(WARNINGS)} warning(s)")
        sys.exit(1)
    print(f"All checks passed. {len(WARNINGS)} warning(s).")


if __name__ == "__main__":
    main()

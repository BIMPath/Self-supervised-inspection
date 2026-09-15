"""
visibility_to_rdf.py — Unity observation conditions into RDF.

Reads the JSON that unity/VisibilityBatchRunner.cs writes and lifts it into evd:ObservationCondition
nodes, together with the evd:VisibilityAnalysis activity that produced them.

The analysis activity is recorded, not just its outputs, because a coverage figure is only
interpretable alongside what the rays were cast against. The same excavation perimeter, analysed
against design BIM geometry alone, will report far higher visibility than when the registered
as-built mesh is included as an occluder — the spoil heaps, plant and stored materials that
actually block the view are not in the design model. Two runs with different occluder sets produce
different, equally valid numbers, and only the recorded provenance says which is which.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PREFIXES = """@prefix sav:  <https://w3id.org/cieo/savigliano#> .
@prefix kb:   <https://w3id.org/cieo/kb#> .
@prefix cieo: <https://w3id.org/cieo/core#> .
@prefix cst:  <https://w3id.org/cieo/construction#> .
@prefix insp: <https://w3id.org/cieo/inspection#> .
@prefix obs:  <https://w3id.org/cieo/observation#> .
@prefix evd:  <https://w3id.org/cieo/evidence#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
"""


def slug(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip())).strip("_")


def esc(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def convert(payload: dict) -> tuple[list[str], dict]:
    analysis_id = f"sav:VisAnalysis_{slug(payload.get('phase_id', 'P0'))}_" \
                  f"{slug(payload.get('generated_at', 'run'))[:15]}"

    out = [
        f"{analysis_id} a evd:VisibilityAnalysis ;",
        f'    evd:analysisEngine "{esc(payload.get("analysis_engine", "Unity"))}" ;',
        f'    evd:analysisVersion "{esc(payload.get("analysis_version", ""))}" ;',
        f'    evd:geometrySource "{esc(payload.get("geometry_source", ""))}" ;',
        f'    evd:occluderSet "{esc(payload.get("occluder_set", ""))}" ;',
        f'    evd:rayCount {int(payload.get("ray_count", 0))} ;',
        f'    rdfs:label "Visibility analysis, {esc(payload.get("scene_name", "scene"))}, '
        f'phase {esc(payload.get("phase_id", ""))}" ;',
        f'    prov:endedAtTime "{esc(payload.get("generated_at", ""))}"^^xsd:dateTime .',
        "",
    ]

    stats = {"conditions": 0, "invalid": 0, "zero_coverage": 0, "targets": set(), "viewpoints": set()}

    for c in payload.get("conditions", []):
        tid, vid = slug(c.get("targetId", "")), slug(c.get("viewpointId", ""))
        if not tid or not vid:
            continue
        stats["conditions"] += 1
        stats["targets"].add(tid)
        stats["viewpoints"].add(vid)

        node = f"sav:Cond_{tid}_{vid}"
        out.append(f"{node} a evd:ObservationCondition ;")
        out.append(f"    evd:assessesTarget sav:{tid} ;")
        out.append(f"    evd:fromViewpoint sav:Viewpoint_{vid} ;")
        out.append(f"    evd:computedBy {analysis_id} ;")
        if c.get("imageId"):
            out.append(f"    evd:conditionOfImage sav:Image_{slug(c['imageId'])} ;")

        out.append(f"    evd:frustumCoverage {float(c.get('frustumCoverage', 0)):.4f} ;")
        out.append(f"    evd:unoccludedFraction {float(c.get('unoccludedFraction', 0)):.4f} ;")
        out.append(f"    evd:effectiveCoverage {float(c.get('effectiveCoverage', 0)):.4f} ;")
        out.append(f"    evd:viewingDistance {float(c.get('viewingDistanceM', 0)):.3f} ;")
        out.append(f"    evd:incidenceAngle {float(c.get('incidenceAngleDeg', 0)):.2f} ;")
        out.append(f"    evd:groundSampleDistance {float(c.get('gsdAlongSurfaceM', 0)):.6f} ;")
        out.append(f"    evd:sampledPointCount {int(c.get('sampledPointCount', 0))} ;")

        valid = bool(c.get("conditionValid", True))
        out.append(f"    evd:conditionValid {'true' if valid else 'false'} ;")
        if not valid:
            stats["invalid"] += 1
            out.append(f'    rdfs:comment "Invalid: {esc(c.get("invalidReason", ""))}" ;')
        if float(c.get("effectiveCoverage", 0)) <= 0:
            stats["zero_coverage"] += 1

        # The per-sample mask travels as a literal so the exact coverage union stays computable
        # from the graph alone, without going back to the Unity run.
        if c.get("visibleSampleMask"):
            out.append(f'    cieo:hasIdentifier "mask:{esc(c["visibleSampleMask"])}" ;')

        out[-1] = out[-1].rstrip(" ;") + " ."

        # Target extent, asserted on the target itself: it is the denominator of every coverage
        # figure and belongs to the target, not to any one viewpoint's view of it.
        if c.get("targetExtent"):
            out.append(f"sav:{tid} cieo:hasSpatialExtent {float(c['targetExtent']):.3f} ;")
            out.append(f'    cieo:extentUnit "{esc(c.get("extentUnit", "m"))}" .')
        out.append("")

    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Unity visibility output into CIEO RDF.")
    ap.add_argument("--conditions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    payload = json.loads(args.conditions.read_text(encoding="utf-8"))
    body, stats = convert(payload)

    header = [
        PREFIXES, "",
        "# Generated by src/visibility_to_rdf.py from the Unity visibility analysis.",
        "# Do not edit by hand.",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(header + body), encoding="utf-8")

    print(f"Wrote {args.out}")
    print(f"  observation conditions  {stats['conditions']}")
    print(f"  distinct targets        {len(stats['targets'])}")
    print(f"  distinct viewpoints     {len(stats['viewpoints'])}")
    print(f"  flagged invalid         {stats['invalid']}")
    print(f"  zero effective coverage {stats['zero_coverage']}")
    if stats["zero_coverage"]:
        print("  (retained deliberately: 'this viewpoint saw none of this target' is evidence "
              "about the capture, and dropping it would reintroduce the silent gap)")


if __name__ == "__main__":
    main()

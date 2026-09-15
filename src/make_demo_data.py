"""
make_demo_data.py — a synthetic Savigliano scene for exercising the pipeline end to end.

WHAT THIS IS NOT
----------------
This is not data, and nothing produced from it is a result. Every number here is invented to
exercise a code path. It exists so that the adapters, the rules and the reasoning can be run and
checked before the real point cloud, the real Unity scene and the real detector outputs are
available, and so that a reader can see the whole chain work without having to reproduce the
reconstruction first.

Replace it with:
  - viewpoints        from the depth-estimation and registration pipeline
  - conditions        from unity/VisibilityBatchRunner.cs run against the phase scenes
  - detections        from the RF-DETR inference run over the selected site imagery

The scene covers one target of each evidential archetype, chosen so that each exercises a
different failure mode:

  ExcavationBoundary_North   PresenceOfRequired    well covered, barrier present
  ExcavationBoundary_South   PresenceOfRequired    poorly covered, barrier not detected
  RoadRestraint_Deck         SpatialCompleteness   covered in three passes, gap in the middle
  BoltedConnection_B12       ComponentIntegrity    close-range, washer not observable
  SiteArea_Whole             VerifiedAbsence       swept thoroughly, nothing found
  Worker_04                  ComponentIntegrity    helmet and vest seen, boots not
"""

from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path("data/demo")

# (target, cieo class, phase, extent, unit, n_samples)
TARGETS = [
    ("ExcavationBoundary_North", "cst:ExcavationBoundary", "P3", 48.0, "m", 64),
    ("ExcavationBoundary_South", "cst:ExcavationBoundary", "P3", 41.0, "m", 64),
    ("RoadRestraint_Deck",       "cst:RoadRestraintSystem", "P6", 122.0, "m", 64),
    ("BoltedConnection_B12",     "cst:BoltedConnection",   "P4", 16.0, "count", 16),
    ("SiteArea_Whole",           "cst:SiteArea",           "P6", 2400.0, "m2", 64),
    ("Worker_04",                "cst:Worker",             "P1", 1.0, "count", 8),
]

# (viewpoint, target, coverage, distance m, incidence deg, unoccluded, visible sample range)
VIEWS = [
    # A well-captured excavation perimeter: six passes, each covering most of the run.
    ("VP_101", "ExcavationBoundary_North", 0.95, 11.0, 14.0, 0.96, (0, 61)),
    ("VP_102", "ExcavationBoundary_North", 0.94, 12.5, 18.0, 0.95, (2, 62)),
    ("VP_103", "ExcavationBoundary_North", 0.92, 13.0, 22.0, 0.94, (3, 62)),
    ("VP_104", "ExcavationBoundary_North", 0.95, 10.0, 12.0, 0.97, (0, 61)),
    ("VP_105", "ExcavationBoundary_North", 0.93, 14.0, 20.0, 0.95, (1, 61)),
    ("VP_106", "ExcavationBoundary_North", 0.96, 9.5, 11.0, 0.97, (0, 62)),

    # The south perimeter: one distant, oblique, half-occluded frame.
    ("VP_110", "ExcavationBoundary_South", 0.12, 44.0, 74.0, 0.38, (0, 8)),

    # Guardrail: three overlapping passes, together covering the whole alignment.
    ("VP_120", "RoadRestraint_Deck", 0.45, 9.0, 16.0, 0.96, (0, 29)),
    ("VP_121", "RoadRestraint_Deck", 0.45, 9.5, 18.0, 0.95, (24, 53)),
    ("VP_122", "RoadRestraint_Deck", 0.31, 10.0, 20.0, 0.94, (48, 64)),

    # Bolted connection: three close, well-exposed looks. Coverage is not the problem here.
    ("VP_130", "BoltedConnection_B12", 0.94, 1.8, 9.0, 0.95, (0, 15)),
    ("VP_131", "BoltedConnection_B12", 0.94, 2.1, 12.0, 0.94, (0, 15)),
    ("VP_132", "BoltedConnection_B12", 0.88, 2.4, 22.0, 0.92, (0, 14)),

    # Site sweep at handover: eight viewpoints over the whole area.
    *[(f"VP_14{i}", "SiteArea_Whole", 0.90, 16.0, 21.0, 0.94, (i * 7, min(64, i * 7 + 58)))
      for i in range(8)],

    # Worker: two frames at working distance.
    ("VP_150", "Worker_04", 0.92, 7.5, 14.0, 0.93, (0, 7)),
    ("VP_151", "Worker_04", 0.88, 9.0, 19.0, 0.90, (0, 7)),
]

# Detections, per viewpoint. Everything not listed becomes an explicit non-detection.
DETECTIONS = {
    "VP_101": [("barrier", 0.93)], "VP_102": [("barrier", 0.91)],
    "VP_103": [("barrier", 0.88)], "VP_104": [("barrier", 0.94)],
    "VP_105": [("barrier", 0.90)], "VP_106": [("barrier", 0.92)],
    # VP_110: nothing. Whether that means the barrier is missing is the question.
    "VP_120": [("barrier", 0.91)], "VP_121": [], "VP_122": [("barrier", 0.89)],
    "VP_130": [("bolt", 0.89), ("nut", 0.86)],
    "VP_131": [("bolt", 0.87), ("nut", 0.88)],
    "VP_132": [("bolt", 0.84), ("nut", 0.82)],
    # No washer from any viewpoint. It was never observable.
    "VP_150": [("helmet", 0.92), ("vest", 0.90)],
    "VP_151": [("helmet", 0.89), ("vest", 0.91)],
    # No boots. Occluded by terrain in both frames.
}

HFOV, WIDTH, HEIGHT = 60.0, 4000, 3000
MODEL = {
    "id": "rf-detr-savigliano/3",
    "architecture": "RF-DETR",
    "confidence_threshold": 0.5,
    "training_set_size": 98,
    "validation_set_size": 28,
    "test_set_size": 14,
    "map": 0.686, "precision": 0.673, "recall": 0.667,
}


def mask(n: int, lo: int, hi: int) -> str:
    v = 0
    for i in range(max(0, lo), min(n, hi)):
        v |= 1 << i
    return format(v, "x")


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    extents = {t[0]: (t[3], t[4], t[5]) for t in TARGETS}

    # -- viewpoints ------------------------------------------------------
    viewpoints = []
    for k, (vid, tid, cov, dist, inc, unocc, _) in enumerate(VIEWS):
        angle = k * 0.7
        viewpoints.append({
            "viewpoint_id": vid,
            "image_id": f"IMG_{vid[3:]}",
            "captured_at": f"2026-04-{14 + (k % 6):02d}T09:{(k * 7) % 60:02d}:00Z",
            "tx": round(dist * math.cos(angle), 3),
            "ty": 1.65, "tz": round(dist * math.sin(angle), 3),
            "qx": 0.0, "qy": round(math.sin(angle / 2), 4), "qz": 0.0,
            "qw": round(math.cos(angle / 2), 4),
            "horizontal_fov_deg": HFOV, "image_width_px": WIDTH, "image_height_px": HEIGHT,
            "pose_source": "estimated", "pose_uncertainty_m": 0.22,
        })
    (OUT / "viewpoints.json").write_text(
        json.dumps({"viewpoints": viewpoints}, indent=2), encoding="utf-8")

    # -- observation conditions (standing in for the Unity run) ----------
    conditions = []
    for vid, tid, cov, dist, inc, unocc, (lo, hi) in VIEWS:
        extent, unit, n = extents[tid]
        gsd_perp = 2 * dist * math.tan(math.radians(HFOV) / 2) / WIDTH
        gsd_along = gsd_perp / max(math.cos(math.radians(inc)), 0.05)
        conditions.append({
            "targetId": tid,
            "targetClass": next(t[1] for t in TARGETS if t[0] == tid),
            "viewpointId": vid,
            "imageId": f"IMG_{vid[3:]}",
            "capturedAt": next(v["captured_at"] for v in viewpoints if v["viewpoint_id"] == vid),
            "targetExtent": extent, "extentUnit": unit,
            "frustumCoverage": round(min(cov / max(unocc, 1e-6), 1.0), 4),
            "unoccludedFraction": unocc,
            "effectiveCoverage": cov,
            "viewingDistanceM": dist,
            "nearestDistanceM": round(dist * 0.85, 2),
            "incidenceAngleDeg": inc,
            "gsdPerpendicularM": round(gsd_perp, 6),
            "gsdAlongSurfaceM": round(gsd_along, 6),
            "sampledPointCount": n,
            "visiblePointCount": max(0, min(n, hi) - max(0, lo)),
            "poseUncertaintyM": 0.22,
            "conditionValid": True,
            "invalidReason": "",
            "visibleSampleMask": mask(n, lo, hi),
        })
    (OUT / "observation_conditions.json").write_text(json.dumps({
        "analysis_engine": "Unity",
        "analysis_version": "cieo-visibility/0.1.0",
        "geometry_source": "design BIM + as-built reconstruction (SYNTHETIC)",
        "occluder_set": "design BIM + registered as-built mesh (SYNTHETIC)",
        "generated_at": "2026-09-10T12:00:00Z",
        "scene_name": "Savigliano_demo",
        "phase_id": "ALL",
        "ray_count": len(conditions),
        "conditions": conditions,
    }, indent=2), encoding="utf-8")

    # -- detections ------------------------------------------------------
    records = []
    for vid, tid, *_ in VIEWS:
        image_id = f"IMG_{vid[3:]}"
        preds = [
            {"class": cls, "confidence": conf,
             "x": 1800 + 40 * i, "y": 1300, "width": 380, "height": 260}
            for i, (cls, conf) in enumerate(DETECTIONS.get(vid, []))
        ]
        records.append({
            "image_id": image_id,
            "image_path": f"data/images/selected/{image_id}.jpg",
            "captured_at": next(v["captured_at"] for v in viewpoints if v["viewpoint_id"] == vid),
            "viewpoint_id": vid,
            "model": MODEL,
            "predictions": preds,
        })
    (OUT / "detections.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    # -- image to zone map, for level-1 grounding ------------------------
    lines = ["image_id,zone_local_name"]
    for vid, tid, *_ in VIEWS:
        lines.append(f"IMG_{vid[3:]},{tid}")
    (OUT / "image_zones.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # -- scene A-Box -----------------------------------------------------
    ttl = [
        "@prefix sav:  <https://w3id.org/cieo/savigliano#> .",
        "@prefix kb:   <https://w3id.org/cieo/kb#> .",
        "@prefix cieo: <https://w3id.org/cieo/core#> .",
        "@prefix cst:  <https://w3id.org/cieo/construction#> .",
        "@prefix insp: <https://w3id.org/cieo/inspection#> .",
        "@prefix obs:  <https://w3id.org/cieo/observation#> .",
        "@prefix owl:  <http://www.w3.org/2002/07/owl#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
        "",
        "# SYNTHETIC scene, generated by src/make_demo_data.py. Not data. Not results.",
        "",
        "sav:SaviglianoBridge a cst:ConstructionProject ;",
        '    rdfs:label "Via Alba bridge replacement, Savigliano"@en ;',
        "    cst:hasPhase kb:P1, kb:P2, kb:P3, kb:P4, kb:P5, kb:P6 .",
        "",
        "sav:Site a cst:ConstructionSite ; rdfs:label \"Savigliano site\"@en .",
        "",
    ]
    for tid, cls, phase, extent, unit, _ in TARGETS:
        ttl.append(f"sav:{tid} a {cls}, insp:InspectionTarget ;")
        ttl.append(f'    rdfs:label "{tid}"@en ;')
        ttl.append(f"    cieo:hasSpatialExtent {extent} ;")
        ttl.append(f'    cieo:extentUnit "{unit}" ;')
        ttl.append(f"    cst:existsDuringPhase kb:{phase} ;")
        ttl.append(f"    cst:belongsToPhase kb:{phase} .")
        ttl.append("")
    for v in viewpoints:
        ttl.append(f"sav:Viewpoint_{v['viewpoint_id']} a obs:Viewpoint ;")
        ttl.append(f"    obs:posX {v['tx']} ; obs:posY {v['ty']} ; obs:posZ {v['tz']} ;")
        ttl.append(f"    obs:quatX {v['qx']} ; obs:quatY {v['qy']} ; "
                   f"obs:quatZ {v['qz']} ; obs:quatW {v['qw']} ;")
        ttl.append(f"    obs:horizontalFov {v['horizontal_fov_deg']} ;")
        ttl.append(f"    obs:imageWidthPx {v['image_width_px']} ; "
                   f"obs:imageHeightPx {v['image_height_px']} ;")
        ttl.append("    obs:hasPoseSource obs:EstimatedPose ;")
        ttl.append(f"    obs:poseUncertainty {v['pose_uncertainty_m']} .")
        ttl.append("")
    Path("data/savigliano-abox.ttl").write_text("\n".join(ttl), encoding="utf-8")

    print(f"Wrote {OUT}/viewpoints.json                ({len(viewpoints)} viewpoints)")
    print(f"Wrote {OUT}/observation_conditions.json    ({len(conditions)} conditions)")
    print(f"Wrote {OUT}/detections.json                ({len(records)} images)")
    print(f"Wrote {OUT}/image_zones.csv")
    print(f"Wrote data/savigliano-abox.ttl             ({len(TARGETS)} targets)")


if __name__ == "__main__":
    build()

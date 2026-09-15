"""
observation_adapter.py — RF-DETR output into RDF observations.

The adapter's job is not just to transcribe detections. Its job is to record the search.

A detector run over an image produces a list of things it found. What it does not produce, and
what no detection format carries, is the list of things it looked for and did not find. In RDF
that gap is fatal: the absence of a detection triple is indistinguishable from the absence of an
attempt to look, and the difference between those two is the difference between non-compliance
and unverifiability.

So for every image, for every class in the active search set, this adapter emits either a
VisualDetection or an explicit NonDetection. Nothing is left to be inferred from silence.

Input format (the shape Roboflow inference returns, one file per image or one combined file):

    {
      "image_id": "IMG_047",
      "image_path": "images/selected/IMG_047.jpg",
      "captured_at": "2026-04-14T09:31:00Z",
      "viewpoint_id": "VP_047",
      "model": {"id": "rf-detr-savigliano/3", "confidence_threshold": 0.5},
      "predictions": [
        {"class": "barrier", "confidence": 0.84, "x": 812, "y": 640, "width": 410, "height": 220}
      ]
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

SAV = "https://w3id.org/cieo/savigliano#"

PREFIXES = """@prefix sav:  <https://w3id.org/cieo/savigliano#> .
@prefix kb:   <https://w3id.org/cieo/kb#> .
@prefix cieo: <https://w3id.org/cieo/core#> .
@prefix cst:  <https://w3id.org/cieo/construction#> .
@prefix insp: <https://w3id.org/cieo/inspection#> .
@prefix obs:  <https://w3id.org/cieo/observation#> .
@prefix evd:  <https://w3id.org/cieo/evidence#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix sosa: <http://www.w3.org/ns/sosa/> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
"""


def slug(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip())).strip("_")


def esc(text: str) -> str:
    return (str(text).replace("\\", "\\\\").replace('"', '\\"')
            .replace("\n", "\\n").replace("\r", ""))


def load_search_set(classes_csv: Path) -> dict[str, str]:
    """
    The classes a non-detection can be recorded for: the detector's vocabulary.

    Duplicate labels in the trained vocabulary are merged here, and the merge is reported.
    A model trained with both 'nut' and 'nuts' will scatter detections across two classes and
    make each look rarer than it is, which corrupts every recall figure downstream.
    """
    mapping, merges = {}, []
    canonical = {"nuts": "nut", "bolt_dup": "bolt"}
    with open(classes_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = row["detector_label"]
            target = canonical.get(label, label)
            if target != label:
                merges.append((label, target))
            mapping[label] = f"kb:Class_{slug(target)}"
    if merges:
        print(f"  vocabulary merges applied: {', '.join(f'{a}->{b}' for a, b in merges)}")
    return mapping


def emit_detector(model: dict) -> list[str]:
    mid = slug(model.get("id", "rf-detr"))
    lines = [
        f"sav:Detector_{mid} a obs:DetectorModel ;",
        f'    rdfs:label "{esc(model.get("id", "RF-DETR"))}" ;',
        f'    obs:modelArchitecture "{esc(model.get("architecture", "RF-DETR"))}" ;',
        f'    obs:modelVersion "{esc(model.get("id", ""))}" ;',
        f'    obs:confidenceThreshold {float(model.get("confidence_threshold", 0.5)):.2f} ;',
    ]
    for key, prop in (("training_set_size", "obs:trainingSetSize"),
                      ("validation_set_size", "obs:validationSetSize"),
                      ("test_set_size", "obs:testSetSize")):
        if key in model:
            lines.append(f"    {prop} {int(model[key])} ;")
    for key, prop in (("map", "obs:reportedMAP"),
                      ("precision", "obs:reportedPrecision"),
                      ("recall", "obs:reportedRecall")):
        if key in model:
            lines.append(f"    {prop} {float(model[key]):.3f} ;")
    lines[-1] = lines[-1].rstrip(" ;") + " ."
    lines.append("")
    return lines


def convert(
    records: list[dict],
    search_set: dict[str, str],
    zone_of_image: dict[str, str] | None = None,
    grounding_method: str = "obs:ZoneAssociation",
    grounding_confidence: float = 0.75,
) -> tuple[list[str], dict[str, int]]:
    """
    Convert detector records into RDF, emitting a non-detection for every class searched and
    not found.

    `zone_of_image` gives a level-1 grounding: which zone each image is attributed to. It is the
    weakest of the three grounding levels and the adapter labels it as such rather than letting a
    coarse association pass for a precise one. Level-2 and level-3 groundings come from the
    visibility analysis, which has the geometry to do better.
    """
    zone_of_image = zone_of_image or {}
    out: list[str] = []
    stats = {"images": 0, "detections": 0, "non_detections": 0, "ungrounded": 0}
    detectors_seen: set[str] = set()

    for rec in records:
        image_id = rec.get("image_id") or Path(rec.get("image_path", "unknown")).stem
        img = f"sav:Image_{slug(image_id)}"
        model = rec.get("model", {})
        did = slug(model.get("id", "rf-detr"))
        detector = f"sav:Detector_{did}"
        threshold = float(model.get("confidence_threshold", 0.5))

        if did not in detectors_seen:
            detectors_seen.add(did)
            out.extend(emit_detector(model))

        stats["images"] += 1
        out.append(f"{img} a obs:SiteImage ;")
        out.append(f'    obs:imagePath "{esc(rec.get("image_path", ""))}" ;')
        if rec.get("captured_at"):
            out.append(f'    obs:capturedAt "{esc(rec["captured_at"])}"^^xsd:dateTime ;')
        if rec.get("viewpoint_id"):
            out.append(f"    obs:takenFrom sav:Viewpoint_{slug(rec['viewpoint_id'])} ;")
        out[-1] = out[-1].rstrip(" ;") + " ."
        out.append("")

        zone = zone_of_image.get(image_id)
        found: set[str] = set()

        for k, pred in enumerate(rec.get("predictions", [])):
            label = pred.get("class")
            class_iri = search_set.get(label)
            if class_iri is None:
                stats["ungrounded"] += 1
                out.append(f"# WARNING: image {image_id} reports class {label!r}, which is not in "
                           f"the declared vocabulary. Detection skipped.")
                continue
            found.add(label)

            det = f"sav:Det_{slug(image_id)}_{k:03d}"
            stats["detections"] += 1
            out.append(f"{det} a obs:VisualDetection ;")
            out.append(f"    obs:detectedClass {class_iri} ;")
            out.append(f"    obs:fromImage {img} ;")
            out.append(f"    obs:producedBy {detector} ;")
            out.append(f'    obs:detectionConfidence {float(pred.get("confidence", 0.0)):.3f} ;')
            out.append(f"    obs:searchThreshold {threshold:.2f} ;")
            for key, prop in (("x", "obs:bboxX"), ("y", "obs:bboxY"),
                              ("width", "obs:bboxWidth"), ("height", "obs:bboxHeight")):
                if key in pred:
                    out.append(f"    {prop} {float(pred[key]):.1f} ;")
            out[-1] = out[-1].rstrip(" ;") + " ."

            if zone:
                g = f"sav:Ground_{slug(image_id)}_{k:03d}"
                out.append(f"{g} a obs:SpatialGrounding ;")
                out.append(f"    obs:groundsObservation {det} ;")
                out.append(f"    obs:groundedTo sav:{zone} ;")
                out.append(f"    obs:usesMethod {grounding_method} ;")
                out.append(f"    obs:groundingConfidence {grounding_confidence:.2f} .")
            else:
                stats["ungrounded"] += 1
                out.append(f"# NOTE: detection {det} has no zone attribution and cannot support "
                           f"any inspection item until it is grounded.")
            out.append("")

        # The part that matters: everything searched for and not found.
        for label, class_iri in search_set.items():
            if label in found:
                continue
            nd = f"sav:NonDet_{slug(image_id)}_{slug(label)}"
            stats["non_detections"] += 1
            out.append(f"{nd} a obs:NonDetection ;")
            out.append(f"    obs:searchedForClass {class_iri} ;")
            out.append(f"    obs:fromImage {img} ;")
            out.append(f"    obs:producedBy {detector} ;")
            out.append(f"    obs:searchThreshold {threshold:.2f} .")
            if zone:
                g = f"sav:Ground_{slug(image_id)}_{slug(label)}_nd"
                out.append(f"{g} a obs:SpatialGrounding ;")
                out.append(f"    obs:groundsObservation {nd} ;")
                out.append(f"    obs:groundedTo sav:{zone} ;")
                out.append(f"    obs:usesMethod {grounding_method} ;")
                out.append(f"    obs:groundingConfidence {grounding_confidence:.2f} .")
            out.append("")

    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert detector output into CIEO RDF observations.")
    ap.add_argument("--detections", required=True, type=Path,
                    help="JSON file: a single record, or a list of records.")
    ap.add_argument("--classes", default=Path("data/object_classes.csv"), type=Path)
    ap.add_argument("--image-zones", type=Path,
                    help="Optional CSV with columns image_id,zone_local_name for level-1 grounding.")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    payload = json.loads(args.detections.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else [payload]

    search_set = load_search_set(args.classes)

    zone_of_image = {}
    if args.image_zones and args.image_zones.exists():
        with open(args.image_zones, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                zone_of_image[row["image_id"]] = row["zone_local_name"]

    body, stats = convert(records, search_set, zone_of_image)

    header = [
        PREFIXES, "",
        "# Generated by src/observation_adapter.py. Do not edit by hand.",
        f'# Generated {datetime.utcnow().isoformat(timespec="seconds")}Z',
        "#",
        "# Every image carries a VisualDetection or a NonDetection for every class in the",
        "# detector's vocabulary. The non-detections are not padding: they are what allow a",
        "# missing barrier to be distinguished from a barrier nobody looked for.",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(header + body), encoding="utf-8")

    print(f"Wrote {args.out}")
    print(f"  images          {stats['images']}")
    print(f"  detections      {stats['detections']}")
    print(f"  non-detections  {stats['non_detections']}")
    if stats["ungrounded"]:
        print(f"  UNGROUNDED      {stats['ungrounded']}  (these support no inspection item)")


if __name__ == "__main__":
    main()

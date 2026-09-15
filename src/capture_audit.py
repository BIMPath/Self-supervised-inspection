"""
capture_audit.py — what can this photo campaign actually settle?

Reads a folder of site photographs, pulls the camera and lens from EXIF, and reports, per object
class, the range beyond which that class falls below the detector's pixel budget. Then it says
which inspection requirements the campaign is capable of settling and which it is not — before a
single detection is run.

This is worth doing first because it is cheap and because it answers a question nobody asks. A
compliance system will happily report "no shear stud detected" on a photograph taken from ten
metres, where a 19 mm stud spans about five pixels. The detector is not wrong and the site is not
non-compliant; the photograph was never capable of settling the question. Knowing the boundary in
metres, per class, turns that from an argument into a measurement.

    python3 src/capture_audit.py --photos "<folder>" --classes data/object_classes.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detectability import ObjectClassSpec, load_object_classes  # noqa: E402

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
except ImportError:  # pragma: no cover
    print("Pillow is required: pip install Pillow")
    raise

# Sensor widths in mm for bodies likely to appear in a construction photo campaign. Add yours if
# it is missing: the horizontal field of view cannot be derived from focal length alone, and
# guessing the sensor size silently scales every range in the report.
SENSOR_WIDTH_MM = {
    "NIKON D80": 23.6, "NIKON D90": 23.6, "NIKON D7000": 23.6, "NIKON D750": 36.0,
    "CANON EOS 70D": 22.5, "CANON EOS 5D MARK III": 36.0,
    "FC3582": 6.3, "FC7303": 6.3, "FC3411": 9.7,        # common DJI camera model codes
    "DEFAULT_APSC": 23.6, "DEFAULT_FULLFRAME": 36.0, "DEFAULT_PHONE": 6.4,
}


def exif_of(path: Path) -> dict:
    try:
        im = Image.open(path)
    except Exception:
        return {}
    width, height = im.size
    base = im.getexif()
    tags = {TAGS.get(k, k): v for k, v in base.items()}
    ifd = base.get_ifd(0x8769)
    tags.update({TAGS.get(k, k): v for k, v in ifd.items()})
    return {
        "width": width,
        "height": height,
        "make": str(tags.get("Make", "")).strip(),
        "model": str(tags.get("Model", "")).strip(),
        "focal_mm": float(tags.get("FocalLength", 0) or 0),
        "focal35_mm": float(tags.get("FocalLengthIn35mmFilm", 0) or 0),
        "datetime": str(tags.get("DateTimeOriginal") or tags.get("DateTime") or ""),
        "has_gps": bool(base.get_ifd(0x8825)),
    }


def sensor_width(model: str, focal_mm: float, focal35_mm: float) -> tuple[float, str]:
    """Sensor width in mm, and how it was determined."""
    key = model.upper()
    if key in SENSOR_WIDTH_MM:
        return SENSOR_WIDTH_MM[key], "known body"
    # Derive from the crop factor when both focal lengths are present: 36 mm / crop.
    if focal_mm > 0 and focal35_mm > 0:
        crop = focal35_mm / focal_mm
        return 36.0 / crop, f"derived from crop factor {crop:.2f}"
    return SENSOR_WIDTH_MM["DEFAULT_APSC"], "ASSUMED APS-C — verify this"


def hfov_deg(focal_mm: float, sensor_w_mm: float) -> float:
    if focal_mm <= 0:
        return 0.0
    return 2 * math.degrees(math.atan(sensor_w_mm / (2 * focal_mm)))


def max_range_m(spec: ObjectClassSpec, hfov: float, px_w: int) -> float:
    """Range beyond which the class falls below its pixel budget, at face-on incidence."""
    if hfov <= 0 or px_w <= 0:
        return 0.0
    gsd_needed = spec.char_dim_installed_m / spec.min_px_on_target
    return gsd_needed * px_w / (2 * math.tan(math.radians(hfov) / 2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit what a photo campaign is capable of settling.")
    ap.add_argument("--photos", required=True, type=Path, help="Folder of photographs (recursive).")
    ap.add_argument("--classes", default=Path("data/object_classes.csv"), type=Path)
    ap.add_argument("--requirements", default=Path("data/inspection_requirements.csv"), type=Path)
    ap.add_argument("--out", type=Path, help="Optional CSV of the per-photo audit.")
    args = ap.parse_args()

    specs = load_object_classes(args.classes)

    photos = [p for p in sorted(args.photos.rglob("*"))
              if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")]
    if not photos:
        print(f"No photographs found under {args.photos}")
        return

    rows, bodies, focals, no_focal, no_gps, undated = [], Counter(), [], 0, 0, 0
    by_session: dict[str, list] = defaultdict(list)

    for p in photos:
        ex = exif_of(p)
        if not ex:
            continue
        sw, how = sensor_width(ex["model"], ex["focal_mm"], ex["focal35_mm"])
        h = hfov_deg(ex["focal_mm"], sw)
        if ex["focal_mm"] <= 0:
            no_focal += 1
        if not ex["has_gps"]:
            no_gps += 1
        if not ex["datetime"]:
            undated += 1
        bodies[f"{ex['make']} {ex['model']}".strip() or "unknown"] += 1
        if ex["focal_mm"] > 0:
            focals.append(ex["focal_mm"])
        row = {"path": str(p.relative_to(args.photos)), "session": p.parent.name,
               "width_px": ex["width"], "focal_mm": ex["focal_mm"], "hfov_deg": round(h, 1),
               "sensor_mm": round(sw, 2), "sensor_source": how, "datetime": ex["datetime"]}
        rows.append(row)
        by_session[p.parent.name].append(row)

    print("=" * 78)
    print(f"CAPTURE AUDIT — {len(rows)} photographs under {args.photos}")
    print("=" * 78)

    print("\nCameras")
    for body, n in bodies.most_common():
        print(f"  {body:<34} {n:5d}")

    print("\nSessions (folders)")
    for s, rs in sorted(by_session.items()):
        fs = [r["focal_mm"] for r in rs if r["focal_mm"] > 0]
        dts = sorted({r["datetime"][:10] for r in rs if r["datetime"]})
        span = f"{min(fs):.0f}-{max(fs):.0f} mm" if fs else "no focal data"
        print(f"  {s:<20} {len(rs):5d} photos   {span:<16} "
              f"dates {', '.join(dts[:2]) if dts else 'none'}")

    print("\nMetadata completeness")
    print(f"  without focal length   {no_focal:5d}  "
          + ("(cannot compute field of view for these)" if no_focal else ""))
    print(f"  without GPS            {no_gps:5d}  "
          + ("(poses must be recovered — by hand in the model, or by reconstruction)"
             if no_gps else ""))
    print(f"  without a timestamp    {undated:5d}")

    if not focals:
        print("\nNo focal lengths in EXIF; cannot compute resolvable ranges.")
        return

    f_wide, f_tele = min(focals), max(focals)
    sw, how = sensor_width(rows[0]["path"] and bodies.most_common(1)[0][0].split(" ", 1)[-1],
                           f_wide, 0)
    sw = rows[0]["sensor_mm"]
    px_w = Counter(r["width_px"] for r in rows).most_common(1)[0][0]
    h_wide, h_tele = hfov_deg(f_wide, sw), hfov_deg(f_tele, sw)

    print(f"\nGeometry used:  {px_w} px wide, sensor {sw:.1f} mm ({rows[0]['sensor_source']})")
    print(f"                {f_wide:.0f} mm -> {h_wide:.1f} deg HFOV    "
          f"{f_tele:.0f} mm -> {h_tele:.1f} deg HFOV")

    print("\n" + "-" * 78)
    print("RESOLVABLE RANGE PER CLASS — beyond this, a non-detection means nothing")
    print("-" * 78)
    print(f"{'class':<16}{'installed dim':>14}{'widest':>11}{'longest':>11}   regime")
    ranked = sorted(specs.values(), key=lambda s: -s.char_dim_installed_m)
    seen = set()
    for s in ranked:
        if s.detector_label in seen or s.detector_label.endswith("_dup"):
            continue
        seen.add(s.detector_label)
        r_wide = max_range_m(s, h_wide, px_w)
        r_tele = max_range_m(s, h_tele, px_w)
        regime = ("scene scale — your photos can settle this" if r_wide >= 20
                  else "component scale — needs close-range capture")
        print(f"{s.detector_label:<16}{s.char_dim_installed_m * 1000:>11.0f} mm"
              f"{r_wide:>10.1f}m{r_tele:>10.1f}m   {regime}")

    # Which requirements the campaign can and cannot reach.
    if args.requirements.exists():
        reach, blocked, manual = [], [], []
        with open(args.requirements, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                needed = [x.strip() for x in (row["required_classes"] or "").split(";") if x.strip()]
                needed += [x.strip() for x in (row["prohibited_classes"] or "").split(";") if x.strip()]
                if not needed:
                    manual.append(row)
                    continue
                worst = min(max_range_m(specs[c], h_wide, px_w) for c in needed if c in specs)
                (reach if worst >= 20 else blocked).append((row, worst))

        print("\n" + "-" * 78)
        print("WHAT THIS CAMPAIGN CAN SETTLE")
        print("-" * 78)
        print(f"\n  Within reach at normal standoff ({len(reach)} requirements):")
        for row, worst in sorted(reach, key=lambda t: -t[1]):
            print(f"    {row['requirement_id']:<12} {row['inspection_item'][:44]:<44} "
                  f"limiting class resolvable to {worst:6.1f} m")

        print(f"\n  Out of reach unless photographed from close range ({len(blocked)}):")
        for row, worst in sorted(blocked, key=lambda t: -t[1]):
            print(f"    {row['requirement_id']:<12} {row['inspection_item'][:44]:<44} "
                  f"needs the camera within {worst:6.2f} m")

        print(f"\n  No visual route at all — documentary, test or missing class ({len(manual)}):")
        for row in manual:
            print(f"    {row['requirement_id']:<12} {row['inspection_item'][:44]}")

        total = len(reach) + len(blocked) + len(manual)
        print(f"\n  {len(reach)}/{total} of the regulatory checklist is reachable by this campaign "
              f"as photographed.")
        print(f"  {len(blocked)}/{total} would become reachable with a close-range capture pass.")
        print(f"  {len(manual)}/{total} will never be settled by photography.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nPer-photo audit written to {args.out}")


if __name__ == "__main__":
    main()

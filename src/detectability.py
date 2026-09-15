"""
detectability.py — the calibrated instrument model R_c(theta).

What this is for
----------------
A detector that reports nothing has told you one of two things: the object is not there, or it
could not have been seen. Nothing in the detector's own output distinguishes them. What does
distinguish them is the expected recall of that detector for that object class under the
observation conditions that actually obtained — how many pixels landed on the object, how much of
it was occluded, at what incidence. That function is R_c(theta), and this module fits it.

Everything downstream depends on R_c being honest. If R is overstated, the framework concludes
"we would have seen it, so it is absent" and reports a violation that may not exist. That failure
is worse than the one the framework is trying to fix, so the module is written to be conservative:
absence conclusions are drawn from the lower confidence bound, never the point estimate, and the
fitted curve refuses to extrapolate silently beyond the conditions it was calibrated on.

Model form
----------
    logit R_c(theta) = logit(R_ref,c)
                       + b1 * log2(p / p_min,c)
                       + b2 * s_c * (1 - v)
                       + b3 * (1 - cos i)

    p       pixels spanning the class's characteristic installed dimension
    p_min,c the class's pixel budget
    v       local unoccluded fraction of the object
    i       incidence angle between view ray and surface normal
    s_c     the class's occlusion sensitivity
    R_ref,c recall measured on held-out data at reference conditions (p = p_min, v = 1, i = 0)

The intercept is anchored to a measured per-class reference recall so that the shared slopes b1,
b2, b3 describe how recall degrades away from that anchor rather than having to carry per-class
level differences as well. With 22 classes and a validation set of realistic size, pooling the
slopes and fitting only the intercepts is what keeps the fit identifiable.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import minimize
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def logit(p: float | np.ndarray) -> float | np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def expit(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))


# ---------------------------------------------------------------------------
# Object classes
# ---------------------------------------------------------------------------

@dataclass
class ObjectClassSpec:
    """The physical and instrument parameters that decide whether a class could be seen."""
    class_id: str
    detector_label: str
    semantic_class: str
    char_dim_installed_m: float
    min_px_on_target: int
    detectability_tier: str
    occlusion_sensitivity: float
    self_occlusion_prior: float
    base_recall_ref: float
    notes: str = ""

    def pixels_on_target(self, gsd_along_surface_m: float) -> float:
        """How many pixels span the characteristic installed dimension at this ground sample distance."""
        if gsd_along_surface_m <= 0:
            return 0.0
        return self.char_dim_installed_m / gsd_along_surface_m

    def max_range_m(self, horizontal_fov_deg: float = 60.0, image_width_px: int = 4000) -> float:
        """
        The range beyond which this class falls below its pixel budget at face-on incidence.

        Useful as a sanity check on a capture plan: if the standoff a site permits is greater than
        this, no amount of photography from that standoff will settle a requirement that depends on
        the class, and the honest thing is to record it as out of automatic scope rather than to
        report non-compliance every time it is not seen.
        """
        gsd_needed = self.char_dim_installed_m / self.min_px_on_target
        return gsd_needed * image_width_px / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))


def load_object_classes(path: str | Path) -> dict[str, ObjectClassSpec]:
    """Load the class vocabulary, merging duplicate detector labels and reporting the merge."""
    specs: dict[str, ObjectClassSpec] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            spec = ObjectClassSpec(
                class_id=row["class_id"],
                detector_label=row["detector_label"],
                semantic_class=row["semantic_class"],
                char_dim_installed_m=float(row["char_dim_installed_m"]),
                min_px_on_target=int(row["min_px_on_target"]),
                detectability_tier=row["detectability_tier"],
                occlusion_sensitivity=float(row["occlusion_sensitivity"]),
                self_occlusion_prior=float(row["self_occlusion_prior"]),
                base_recall_ref=float(row["base_recall_ref"]),
                notes=row.get("notes", ""),
            )
            specs[spec.detector_label] = spec
    return specs


# ---------------------------------------------------------------------------
# Observation conditions
# ---------------------------------------------------------------------------

@dataclass
class ObservationCondition:
    """
    One target seen from one viewpoint: the output of the visibility analysis in the virtual
    environment. Field names match the JSON the Unity batch runner writes.
    """
    target_id: str
    viewpoint_id: str
    image_id: str = ""
    captured_at: str = ""
    target_extent: float = 0.0
    extent_unit: str = "m"
    frustum_coverage: float = 0.0
    unoccluded_fraction: float = 0.0
    effective_coverage: float = 0.0
    viewing_distance_m: float = 0.0
    incidence_angle_deg: float = 0.0
    gsd_perpendicular_m: float = 0.0
    gsd_along_surface_m: float = 0.0
    sampled_point_count: int = 0
    visible_point_count: int = 0
    pose_uncertainty_m: float = 0.0
    condition_valid: bool = True
    invalid_reason: str = ""
    visible_sample_mask: str = ""   # hex bitmask over the target's deterministic sample set

    @classmethod
    def from_unity(cls, d: dict) -> "ObservationCondition":
        """Build from the camelCase JSON Unity's JsonUtility emits."""
        return cls(
            target_id=d.get("targetId", ""),
            viewpoint_id=d.get("viewpointId", ""),
            image_id=d.get("imageId", ""),
            captured_at=d.get("capturedAt", ""),
            target_extent=float(d.get("targetExtent", 0.0)),
            extent_unit=d.get("extentUnit", "m"),
            frustum_coverage=float(d.get("frustumCoverage", 0.0)),
            unoccluded_fraction=float(d.get("unoccludedFraction", 0.0)),
            effective_coverage=float(d.get("effectiveCoverage", 0.0)),
            viewing_distance_m=float(d.get("viewingDistanceM", 0.0)),
            incidence_angle_deg=float(d.get("incidenceAngleDeg", 0.0)),
            gsd_perpendicular_m=float(d.get("gsdPerpendicularM", 0.0)),
            gsd_along_surface_m=float(d.get("gsdAlongSurfaceM", 0.0)),
            sampled_point_count=int(d.get("sampledPointCount", 0)),
            visible_point_count=int(d.get("visiblePointCount", 0)),
            pose_uncertainty_m=float(d.get("poseUncertaintyM", 0.0)),
            condition_valid=bool(d.get("conditionValid", True)),
            invalid_reason=d.get("invalidReason", ""),
            visible_sample_mask=d.get("visibleSampleMask", ""),
        )

    def sample_mask_array(self) -> np.ndarray | None:
        """
        Decode the per-sample visibility bitmask into a boolean array.

        The mask is what makes exact coverage unions possible. Because the target's sampling is
        deterministic in its seed, sample index i denotes the same physical point in every
        viewpoint's record, so the union of two viewpoints' coverage is a bitwise OR rather than
        an assumption about how their coverage fractions overlap. Guessing that overlap is how a
        system convinces itself it has seen a whole perimeter it has only seen one end of, twice.
        """
        if not self.visible_sample_mask or self.sampled_point_count <= 0:
            return None
        try:
            raw = int(self.visible_sample_mask, 16)
        except ValueError:
            return None
        return np.array(
            [(raw >> i) & 1 for i in range(self.sampled_point_count)], dtype=bool
        )


# ---------------------------------------------------------------------------
# The detectability model
# ---------------------------------------------------------------------------

@dataclass
class DetectabilityModel:
    """
    R_c(theta), with pooled slopes and per-class anchored intercepts.

    Default slopes are placeholders, deliberately labelled as such. They produce sensible
    behaviour but they are not evidence, and `calibrated` stays False until `fit` has run on real
    data. Any result carried into a paper should come from a fitted model, and the flag exists so
    that an uncalibrated run cannot be mistaken for a calibrated one downstream.
    """
    b_resolution: float = 0.90     # per doubling of pixels on target
    b_occlusion: float = -4.00     # scaled by the class's occlusion sensitivity
    b_incidence: float = -2.00     # per unit of (1 - cos incidence)

    calibrated: bool = False
    n_calibration: int = 0
    calibration_classes: list[str] = field(default_factory=list)
    slope_ci: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Conditions outside the calibrated envelope are reported rather than silently extrapolated.
    px_range: tuple[float, float] = (0.0, 0.0)
    incidence_range: tuple[float, float] = (0.0, 0.0)

    # ---- evaluation ----

    def expected_recall(
        self,
        spec: ObjectClassSpec,
        cond: ObservationCondition,
        local_visibility: float | None = None,
    ) -> float:
        """
        Expected recall for one class under one observation condition.

        `local_visibility` is the unoccluded fraction of the object itself. It defaults to the
        target's unoccluded fraction, which is a reasonable proxy when the object is large
        relative to the target and a poor one when it is not.
        """
        if not cond.condition_valid or cond.gsd_along_surface_m <= 0:
            return 0.0

        p = spec.pixels_on_target(cond.gsd_along_surface_m)
        if p <= 0:
            return 0.0

        v = cond.unoccluded_fraction if local_visibility is None else local_visibility
        v = float(np.clip(v, 0.0, 1.0))
        cos_i = math.cos(math.radians(min(cond.incidence_angle_deg, 89.9)))

        z = (
            logit(spec.base_recall_ref)
            + self.b_resolution * math.log2(max(p / spec.min_px_on_target, 1e-6))
            + self.b_occlusion * spec.occlusion_sensitivity * (1.0 - v)
            + self.b_incidence * (1.0 - cos_i)
        )
        return float(expit(z))

    def recall_lower_bound(
        self,
        spec: ObjectClassSpec,
        cond: ObservationCondition,
        local_visibility: float | None = None,
        z: float = 1.645,   # one-sided 95%
    ) -> float:
        """
        Conservative recall, for use in absence reasoning.

        Uses the lower end of the fitted slope confidence intervals, so that a claim of the form
        "we would have detected it" rests on the least favourable defensible reading of the
        calibration rather than the most favourable. Falls back to a flat 20% haircut when the
        model is uncalibrated, which keeps uncalibrated runs from ever looking authoritative.
        """
        point = self.expected_recall(spec, cond, local_visibility)
        if not self.calibrated or not self.slope_ci:
            return float(np.clip(point * 0.8, 0.0, 1.0))

        conservative = DetectabilityModel(
            b_resolution=self.slope_ci.get("b_resolution", (self.b_resolution,) * 2)[0],
            b_occlusion=self.slope_ci.get("b_occlusion", (self.b_occlusion,) * 2)[0],
            b_incidence=self.slope_ci.get("b_incidence", (self.b_incidence,) * 2)[0],
        )
        return float(min(point, conservative.expected_recall(spec, cond, local_visibility)))

    def within_envelope(self, spec: ObjectClassSpec, cond: ObservationCondition) -> bool:
        """True when this condition lies inside the range the model was calibrated over."""
        if not self.calibrated:
            return False
        p = spec.pixels_on_target(cond.gsd_along_surface_m)
        return (
            self.px_range[0] <= p <= self.px_range[1]
            and self.incidence_range[0] <= cond.incidence_angle_deg <= self.incidence_range[1]
        )

    # ---- calibration ----

    def fit(
        self,
        samples: list[tuple[ObjectClassSpec, ObservationCondition, bool]],
        n_bootstrap: int = 400,
        seed: int = 20260910,
    ) -> "DetectabilityModel":
        """
        Fit the pooled slopes by maximum likelihood, with bootstrap confidence intervals.

        `samples` is a list of (class spec, observation condition, was_detected) for ground-truth
        instances that were PRESENT. Only present instances belong here: recall is conditional on
        presence, and including absent ones would fit something else entirely.
        """
        if not samples:
            raise ValueError("no calibration samples supplied")
        if not _HAVE_SCIPY:
            raise RuntimeError("scipy is required to fit the detectability model")

        X, y = self._design_matrix(samples)

        def nll(beta: np.ndarray) -> float:
            z = X[:, 0] + X[:, 1] * beta[0] + X[:, 2] * beta[1] + X[:, 3] * beta[2]
            p = expit(z)
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

        start = np.array([self.b_resolution, self.b_occlusion, self.b_incidence])
        res = minimize(nll, start, method="Nelder-Mead",
                       options={"maxiter": 6000, "xatol": 1e-6, "fatol": 1e-6})
        self.b_resolution, self.b_occlusion, self.b_incidence = (float(v) for v in res.x)

        # Bootstrap the slopes. Preferred over a Wald interval here because the design is
        # unbalanced across classes and the asymptotic assumption is not safe at these sizes.
        rng = np.random.default_rng(seed)
        n = len(samples)
        draws = np.empty((n_bootstrap, 3))
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            Xb, yb = X[idx], y[idx]

            def nll_b(beta: np.ndarray, Xb=Xb, yb=yb) -> float:
                z = Xb[:, 0] + Xb[:, 1] * beta[0] + Xb[:, 2] * beta[1] + Xb[:, 3] * beta[2]
                p = np.clip(expit(z), 1e-9, 1 - 1e-9)
                return float(-np.sum(yb * np.log(p) + (1 - yb) * np.log(1 - p)))

            rb = minimize(nll_b, res.x, method="Nelder-Mead",
                          options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4})
            draws[b] = rb.x

        lo, hi = np.percentile(draws, [5, 95], axis=0)
        self.slope_ci = {
            "b_resolution": (float(lo[0]), float(hi[0])),
            "b_occlusion": (float(lo[1]), float(hi[1])),
            "b_incidence": (float(lo[2]), float(hi[2])),
        }

        px = np.array([s.pixels_on_target(c.gsd_along_surface_m) for s, c, _ in samples])
        inc = np.array([c.incidence_angle_deg for _, c, _ in samples])
        self.px_range = (float(px.min()), float(px.max()))
        self.incidence_range = (float(inc.min()), float(inc.max()))

        self.calibrated = True
        self.n_calibration = n
        self.calibration_classes = sorted({s.detector_label for s, _, _ in samples})
        return self

    @staticmethod
    def _design_matrix(samples) -> tuple[np.ndarray, np.ndarray]:
        rows, ys = [], []
        for spec, cond, detected in samples:
            p = spec.pixels_on_target(cond.gsd_along_surface_m)
            v = float(np.clip(cond.unoccluded_fraction, 0.0, 1.0))
            cos_i = math.cos(math.radians(min(cond.incidence_angle_deg, 89.9)))
            rows.append([
                logit(spec.base_recall_ref),                    # anchored offset
                math.log2(max(p / spec.min_px_on_target, 1e-6)),
                spec.occlusion_sensitivity * (1.0 - v),
                1.0 - cos_i,
            ])
            ys.append(1.0 if detected else 0.0)
        return np.asarray(rows, dtype=float), np.asarray(ys, dtype=float)

    # ---- serialisation ----

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "DetectabilityModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["slope_ci"] = {k: tuple(v) for k, v in d.get("slope_ci", {}).items()}
        d["px_range"] = tuple(d.get("px_range", (0.0, 0.0)))
        d["incidence_range"] = tuple(d.get("incidence_range", (0.0, 0.0)))
        return cls(**d)


# ---------------------------------------------------------------------------
# Precision calibration — detector confidence is not precision
# ---------------------------------------------------------------------------

@dataclass
class PrecisionCalibration:
    """
    Maps a detector confidence score to the probability that the detection is a true positive.

    These are not the same quantity, and treating them as if they were is one of the specific
    errors this framework sets out to avoid. A detector reporting 0.87 is not saying that 87% of
    its 0.87-confidence detections are correct; what fraction are correct is an empirical property
    of the model and the domain, measured by binning held-out detections and counting.
    """
    bin_edges: list[float] = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    precision_by_bin: list[float] = field(default_factory=lambda: [0.55, 0.65, 0.74, 0.82, 0.90])
    n_by_bin: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    calibrated: bool = False

    def precision(self, confidence: float) -> float:
        for i in range(len(self.bin_edges) - 1):
            if self.bin_edges[i] <= confidence < self.bin_edges[i + 1]:
                return self.precision_by_bin[i]
        return self.precision_by_bin[-1] if confidence >= self.bin_edges[-1] else 0.0

    def fit(self, detections: list[tuple[float, bool]]) -> "PrecisionCalibration":
        """`detections` is (confidence, is_true_positive) over a held-out set."""
        prec, counts = [], []
        for i in range(len(self.bin_edges) - 1):
            lo, hi = self.bin_edges[i], self.bin_edges[i + 1]
            binned = [tp for c, tp in detections if lo <= c < hi]
            counts.append(len(binned))
            # Laplace smoothing keeps an empty or tiny bin from asserting 0.0 or 1.0.
            prec.append((sum(binned) + 1) / (len(binned) + 2) if binned else float("nan"))

        # Carry the last populated estimate forward through empty bins rather than leaving NaN.
        last = 0.5
        filled = []
        for p in prec:
            if not math.isnan(p):
                last = p
            filled.append(last)

        self.precision_by_bin = filled
        self.n_by_bin = counts
        self.calibrated = True
        return self

    def expected_calibration_error(self, detections: list[tuple[float, bool]]) -> float:
        """Standard ECE, reported so the gap between confidence and precision is quantified."""
        if not detections:
            return float("nan")
        total, n = 0.0, len(detections)
        for i in range(len(self.bin_edges) - 1):
            lo, hi = self.bin_edges[i], self.bin_edges[i + 1]
            binned = [(c, tp) for c, tp in detections if lo <= c < hi]
            if not binned:
                continue
            mean_conf = sum(c for c, _ in binned) / len(binned)
            acc = sum(1 for _, tp in binned if tp) / len(binned)
            total += len(binned) / n * abs(mean_conf - acc)
        return total

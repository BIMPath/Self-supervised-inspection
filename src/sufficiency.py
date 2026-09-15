"""
sufficiency.py — is the evidence capable of settling this question?

The framework asks two questions in order, and the order matters.

    1. Could this evidence settle the question?      (this module)
    2. What does the evidence say?                   (decision.py)

Answering them together is how a system ends up reporting a violation whenever a camera happened
to point the wrong way. Answering them separately means the second question is only ever put when
the first has been answered yes, and the honest outcome when it has not is to decline.

Three models are implemented, so that the value of the framework's approach can be measured
against what it replaces rather than asserted:

    NaiveClosedWorld    any non-detection is absence. The implicit model behind detection-driven
                        compliance checking. Always settles, and is wrong whenever the target was
                        not properly observed.

    HeuristicScore      a weighted combination of confidence, visibility, coverage, occlusion and
                        temporal validity against fixed thresholds. Evidence-aware in form, but
                        the thresholds are asserted rather than calibrated.

    BayesianAbsence     the posterior probability that the condition truly does not hold, given
                        that nothing was detected, computed from calibrated per-viewpoint
                        detectability aggregated across every viewpoint that observed the target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from detectability import (
    DetectabilityModel,
    ObjectClassSpec,
    ObservationCondition,
    PrecisionCalibration,
)


class SufficiencyLevel(str, Enum):
    SUFFICIENT = "Sufficient"
    MARGINAL = "Marginal"
    INSUFFICIENT = "Insufficient"


class SufficiencyModelKind(str, Enum):
    BAYESIAN_ABSENCE = "BayesianAbsenceModel"
    HEURISTIC_SCORE = "HeuristicScoreModel"
    NAIVE_CLOSED_WORLD = "NaiveClosedWorldModel"


@dataclass
class SufficiencyThresholds:
    """
    Decision thresholds. Every one of these must be calibrated on the project's own data before it
    appears in a result; the defaults here are starting points for that calibration, not findings.

    `posterior_absence_high` is the one to be most careful with. It governs how confident the
    system must be before it will assert that something regulatory is missing, and at a prior of
    0.8 it implies an aggregate detection probability of about 0.98 — demanding by design, because
    a false accusation of non-compliance is a more expensive error than an abstention.
    """
    posterior_absence_high: float = 0.90     # settle as non-compliant above this
    posterior_absence_low: float = 0.60      # below this, the evidence taught us too little
    positive_confidence_high: float = 0.85   # settle as compliant above this
    positive_confidence_low: float = 0.60
    required_coverage_default: float = 0.95  # for spatial completeness
    marginal_coverage: float = 0.70
    heuristic_high: float = 0.75             # the README's E >= 0.75
    heuristic_low: float = 0.40              # the README's E < 0.40
    evidence_half_life_days: float = 3.0     # site conditions are not persistent


@dataclass
class SufficiencyResult:
    """The verdict, with every intermediate quantity kept so the reasoning can be shown."""
    item_id: str
    model: SufficiencyModelKind
    level: SufficiencyLevel

    aggregated_recall: float = 0.0
    aggregated_recall_lower: float = 0.0
    prior_presence: float = 0.8
    posterior_absence: float = 0.0
    posterior_shift: float = 0.0
    coverage_achieved: float = 0.0
    coverage_required: float = 0.0
    heuristic_score: float = 0.0
    temporal_validity: float = 1.0
    evidence_age_days: float = 0.0

    limiting_class: str = ""
    n_viewpoints: int = 0
    n_valid_viewpoints: int = 0
    outside_envelope: bool = False
    rationale: str = ""
    recommended_action: str = ""
    per_viewpoint_detection: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-viewpoint detection probability
# ---------------------------------------------------------------------------

def viewpoint_detection_probability(
    spec: ObjectClassSpec,
    cond: ObservationCondition,
    model: DetectabilityModel,
    conservative: bool = True,
) -> float:
    """
    Probability that this viewpoint would have detected an instance of `spec` on the target,
    given that one is present.

    Three independent things all have to go right, and the product is the point:

        effective_coverage      the object's location fell inside the frustum and was not
                                occluded by anything else in the scene
        (1 - self_occlusion)    the object was not hidden by the assembly it belongs to
        R_c(theta)              the detector resolved it at that range and incidence

    A washer illustrates why all three are needed. It can be six metres away, dead centre of a
    perfectly exposed frame — effective coverage 1.0 — and still be unobservable, because once the
    nut is torqued down the washer is behind it. The geometry term says the location was seen; the
    self-occlusion term says the object was not.
    """
    if not cond.condition_valid:
        return 0.0

    recall = (
        model.recall_lower_bound(spec, cond) if conservative
        else model.expected_recall(spec, cond)
    )
    return float(
        np.clip(cond.effective_coverage, 0.0, 1.0)
        * (1.0 - spec.self_occlusion_prior)
        * recall
    )


def aggregate_detection_probability(per_viewpoint: list[float]) -> float:
    """
    Probability that at least one viewpoint would have detected the object: 1 - prod(1 - d_v).

    The independence assumption behind the product is an approximation, and an optimistic one.
    Viewpoints along a single walked path share illumination, weather, and very often the same
    occluders, so their failures are correlated and the true aggregate is lower than this. The
    limitation is stated rather than hidden; `effective_independent_views` below gives a blunt
    way to discount for it.
    """
    q = 1.0
    for d in per_viewpoint:
        q *= (1.0 - float(np.clip(d, 0.0, 1.0)))
    return float(1.0 - q)


def aggregate_detection_probability_correlated(
    per_viewpoint: list[float], correlation: float = 0.3
) -> float:
    """
    A discounted aggregate for correlated viewpoints.

    Down-weights each additional viewpoint's independent contribution by (1 - correlation),
    so that twenty photographs from one standing position do not add up to certainty. Crude, but
    it moves the error in the safe direction, which an unqualified independence assumption does
    not.
    """
    if not per_viewpoint:
        return 0.0
    ordered = sorted(per_viewpoint, reverse=True)
    q = 1.0 - ordered[0]
    for k, d in enumerate(ordered[1:], start=1):
        q *= (1.0 - d * (1.0 - correlation) ** k)
    return float(1.0 - q)


def effective_independent_views(per_viewpoint: list[float], correlation: float = 0.3) -> float:
    """Number of independent looks the viewpoint set is worth, for reporting."""
    if not per_viewpoint:
        return 0.0
    return float(1.0 + (len(per_viewpoint) - 1) * (1.0 - correlation))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def union_coverage(conditions: list[ObservationCondition]) -> tuple[float, bool]:
    """
    Fraction of the target's extent observed by at least one viewpoint.

    Returns (coverage, exact). When every condition carries a per-sample visibility bitmask the
    union is computed exactly by bitwise OR, because sample index i denotes the same physical
    point across viewpoints. Without masks it falls back to an inclusion-exclusion style bound
    that assumes the worst case of maximal overlap — deliberately pessimistic, because the
    alternative assumption, that separate viewpoints see disjoint parts of the target, is how a
    system talks itself into having covered a perimeter it saw one end of twice.
    """
    valid = [c for c in conditions if c.condition_valid]
    if not valid:
        return 0.0, True

    masks = [c.sample_mask_array() for c in valid]
    if all(m is not None for m in masks) and len({len(m) for m in masks}) == 1:
        union = np.zeros_like(masks[0], dtype=bool)
        for m in masks:
            union |= m
        return float(union.mean()), True

    return float(max(c.effective_coverage for c in valid)), False


def temporal_validity(age_days: float, half_life_days: float) -> float:
    """
    Exponential decay of evidential weight with age.

    A barrier photographed on Monday is weak evidence about Thursday, and a system that treats
    all evidence as current will certify conditions that no longer hold. The half-life should be
    set from how fast the site actually changes during the phase in question, not chosen once for
    the whole project.
    """
    if age_days <= 0:
        return 1.0
    return float(math.exp(-math.log(2) * age_days / max(half_life_days, 1e-6)))


# ---------------------------------------------------------------------------
# Bayesian absence
# ---------------------------------------------------------------------------

def posterior_absence(
    aggregated_recall: float,
    prior_presence: float,
    false_positive_rate: float = 0.0,
) -> float:
    """
    P(condition does not hold | nothing was detected).

                             (1 - FPR)(1 - pi)
        P(absent | not d) = -------------------------------------
                            (1 - FPR)(1 - pi) + (1 - D) pi

    where D is the aggregate probability of detecting the condition had it held, and pi the prior
    that it holds.

    The behaviour at the limits is what makes this the right quantity. As D approaches 1, the
    posterior approaches 1: we looked properly and saw nothing, so it is absent. As D approaches
    0, the posterior collapses back to the prior 1 - pi: we learned nothing, and the system says
    so rather than defaulting to an accusation.
    """
    pi = float(np.clip(prior_presence, 1e-6, 1 - 1e-6))
    d = float(np.clip(aggregated_recall, 0.0, 1.0))
    fpr = float(np.clip(false_positive_rate, 0.0, 1 - 1e-6))

    num = (1.0 - fpr) * (1.0 - pi)
    den = num + (1.0 - d) * pi
    return float(num / den) if den > 0 else 1.0


def posterior_shift(posterior: float, prior_presence: float) -> float:
    """
    How far the evidence moved belief from the prior toward certainty of absence, in [0, 1].

    Zero means the observation was uninformative — which is a finding about the capture, not about
    the site, and is exactly what should be reported back to whoever planned the photography.
    """
    prior_absent = 1.0 - float(np.clip(prior_presence, 1e-6, 1 - 1e-6))
    if 1.0 - prior_absent <= 1e-9:
        return 0.0
    return float(np.clip((posterior - prior_absent) / (1.0 - prior_absent), 0.0, 1.0))


# ---------------------------------------------------------------------------
# The heuristic baseline
# ---------------------------------------------------------------------------

def heuristic_score(
    detection_confidence: float,
    visibility: float,
    spatial_coverage: float,
    occlusion: float,
    temporal: float,
    weights: tuple[float, float, float, float, float] = (0.30, 0.20, 0.25, 0.15, 0.10),
) -> float:
    """
    E = f(C, V, S, O, T) — the weighted score from the project's original plan, kept as the
    ablation baseline.

    It is evidence-aware in form: it does look at visibility, coverage and occlusion rather than
    at detector confidence alone. What it lacks is any connection between those quantities and the
    detector's actual behaviour. The weights and the 0.75 / 0.40 cut-offs are asserted, so the
    score has no units and no interpretation — 0.62 does not mean anything about how likely the
    system is to be wrong. Comparing it against the calibrated model is what shows whether
    calibration buys anything, or whether being roughly evidence-aware is enough.
    """
    terms = (detection_confidence, visibility, spatial_coverage, 1.0 - occlusion, temporal)
    return float(np.clip(sum(w * float(np.clip(t, 0.0, 1.0)) for w, t in zip(weights, terms)), 0.0, 1.0))


# ---------------------------------------------------------------------------
# The assessor
# ---------------------------------------------------------------------------

class SufficiencyAssessor:
    def __init__(
        self,
        detectability: DetectabilityModel,
        thresholds: SufficiencyThresholds | None = None,
        precision: PrecisionCalibration | None = None,
        correlation: float = 0.3,
    ):
        self.detectability = detectability
        self.t = thresholds or SufficiencyThresholds()
        self.precision = precision or PrecisionCalibration()
        self.correlation = correlation

    # -- Bayesian ----------------------------------------------------------

    def assess_bayesian(
        self,
        item_id: str,
        required_classes: list[ObjectClassSpec],
        conditions: list[ObservationCondition],
        prior_presence: float = 0.8,
        evidence_age_days: float = 0.0,
        required_coverage: float | None = None,
        conjunctive: bool = False,
    ) -> SufficiencyResult:
        """
        Assess whether the evidence can settle an item.

        With `conjunctive=True` (component integrity: bolt AND nut AND washer; helmet AND vest AND
        boots) the aggregate is bounded by the least detectable component, and that component is
        reported as the limiting factor. Naming it is what turns an unhelpful verdict into an
        instruction: 'insufficient evidence' tells an inspector nothing, 'the washer could not have
        been seen from any of these viewpoints, photograph the connection from the nut side'
        tells them what to do.
        """
        valid = [c for c in conditions if c.condition_valid]
        result = SufficiencyResult(
            item_id=item_id,
            model=SufficiencyModelKind.BAYESIAN_ABSENCE,
            level=SufficiencyLevel.INSUFFICIENT,
            prior_presence=prior_presence,
            n_viewpoints=len(conditions),
            n_valid_viewpoints=len(valid),
            evidence_age_days=evidence_age_days,
        )

        if not valid or not required_classes:
            result.rationale = (
                "No valid observation conditions for this target. The target was either never "
                "inside a camera frustum, or its geometry is missing from the scene."
            )
            result.recommended_action = "Capture imagery of this target, or add its geometry to the model."
            return result

        cov, exact = union_coverage(valid)
        result.coverage_achieved = cov
        # Zero, not the default, when the caller did not ask for a coverage requirement. Only
        # spatial-completeness items are predicated over an extent; applying a coverage veto to a
        # bolted connection would report a resolution problem as a coverage problem and send the
        # inspector to photograph more of the connection rather than closer to it.
        result.coverage_required = required_coverage if required_coverage is not None else 0.0

        tv = temporal_validity(evidence_age_days, self.t.evidence_half_life_days)
        result.temporal_validity = tv

        # Per-class aggregate detection probability across viewpoints.
        per_class: dict[str, float] = {}
        per_class_vp: dict[str, list[float]] = {}
        for spec in required_classes:
            dv = [viewpoint_detection_probability(spec, c, self.detectability) for c in valid]
            per_class_vp[spec.detector_label] = dv
            per_class[spec.detector_label] = aggregate_detection_probability_correlated(
                dv, self.correlation
            )

        if conjunctive:
            # Bounded by the weakest component.
            limiting = min(per_class, key=per_class.get)
            aggregate = per_class[limiting]
            result.limiting_class = limiting
            result.per_viewpoint_detection = per_class_vp[limiting]
        else:
            # Any of the acceptable classes will do, so take the best.
            best = max(per_class, key=per_class.get)
            aggregate = per_class[best]
            result.limiting_class = best
            result.per_viewpoint_detection = per_class_vp[best]

        aggregate *= tv
        result.aggregated_recall = aggregate
        result.aggregated_recall_lower = aggregate  # already conservative: lower-bound recall used

        post = posterior_absence(aggregate, prior_presence)
        result.posterior_absence = post
        result.posterior_shift = posterior_shift(post, prior_presence)

        result.outside_envelope = any(
            not self.detectability.within_envelope(spec, c)
            for spec in required_classes for c in valid
        )

        # Level.
        if post >= self.t.posterior_absence_high:
            result.level = SufficiencyLevel.SUFFICIENT
        elif post >= self.t.posterior_absence_low:
            result.level = SufficiencyLevel.MARGINAL
        else:
            result.level = SufficiencyLevel.INSUFFICIENT

        # Coverage can veto: a spatial-completeness item is not settled by a confident look at
        # a fraction of the extent, however good that look was.
        if required_coverage is not None and cov < required_coverage:
            if cov < self.t.marginal_coverage:
                result.level = SufficiencyLevel.INSUFFICIENT
            elif result.level == SufficiencyLevel.SUFFICIENT:
                result.level = SufficiencyLevel.MARGINAL

        result.rationale = self._explain(result, exact, conjunctive)
        result.recommended_action = self._recommend(result, required_classes, valid)
        return result

    # -- Heuristic ---------------------------------------------------------

    def assess_heuristic(
        self,
        item_id: str,
        conditions: list[ObservationCondition],
        detection_confidence: float,
        evidence_age_days: float = 0.0,
    ) -> SufficiencyResult:
        valid = [c for c in conditions if c.condition_valid]
        result = SufficiencyResult(
            item_id=item_id,
            model=SufficiencyModelKind.HEURISTIC_SCORE,
            level=SufficiencyLevel.INSUFFICIENT,
            n_viewpoints=len(conditions),
            n_valid_viewpoints=len(valid),
            evidence_age_days=evidence_age_days,
        )
        if not valid:
            result.rationale = "No valid observation conditions."
            return result

        best = max(valid, key=lambda c: c.effective_coverage)
        cov, _ = union_coverage(valid)
        tv = temporal_validity(evidence_age_days, self.t.evidence_half_life_days)

        e = heuristic_score(
            detection_confidence=detection_confidence,
            visibility=best.unoccluded_fraction,
            spatial_coverage=cov,
            occlusion=1.0 - best.unoccluded_fraction,
            temporal=tv,
        )
        result.heuristic_score = e
        result.coverage_achieved = cov
        result.temporal_validity = tv

        if e >= self.t.heuristic_high:
            result.level = SufficiencyLevel.SUFFICIENT
        elif e >= self.t.heuristic_low:
            result.level = SufficiencyLevel.MARGINAL
        else:
            result.level = SufficiencyLevel.INSUFFICIENT

        result.rationale = f"Heuristic evidence score E = {e:.2f} against fixed cut-offs " \
                           f"{self.t.heuristic_low:.2f} / {self.t.heuristic_high:.2f}."
        return result

    # -- Naive -------------------------------------------------------------

    def assess_naive(self, item_id: str, conditions: list[ObservationCondition]) -> SufficiencyResult:
        """Always sufficient. That is the assumption under test, not an oversight."""
        return SufficiencyResult(
            item_id=item_id,
            model=SufficiencyModelKind.NAIVE_CLOSED_WORLD,
            level=SufficiencyLevel.SUFFICIENT,
            aggregated_recall=1.0,
            posterior_absence=1.0,
            n_viewpoints=len(conditions),
            n_valid_viewpoints=len([c for c in conditions if c.condition_valid]),
            rationale="Closed-world assumption: any non-detection is treated as absence.",
        )

    # -- Explanation -------------------------------------------------------

    def _explain(self, r: SufficiencyResult, exact_coverage: bool, conjunctive: bool) -> str:
        parts = [
            f"Across {r.n_valid_viewpoints} valid viewpoint(s), the aggregate probability of "
            f"detecting the required condition had it held is {r.aggregated_recall:.2f}."
        ]
        if conjunctive and r.limiting_class:
            parts.append(
                f"The assembly is conjunctive and '{r.limiting_class}' is the least observable "
                f"component, so it bounds the whole."
            )
        parts.append(
            f"Given a prior of {r.prior_presence:.2f} that the condition holds, a non-detection "
            f"puts the probability of true absence at {r.posterior_absence:.2f} "
            f"(prior was {1 - r.prior_presence:.2f}; shift {r.posterior_shift:.2f})."
        )
        parts.append(
            f"Observed coverage of the target extent is {r.coverage_achieved:.0%}"
            + ("." if exact_coverage else ", estimated without per-sample masks and therefore a lower bound.")
        )
        if r.temporal_validity < 0.95:
            parts.append(
                f"Evidence is {r.evidence_age_days:.1f} days old, discounting its weight to "
                f"{r.temporal_validity:.2f}."
            )
        if r.outside_envelope:
            parts.append(
                "WARNING: at least one observation condition lies outside the range the "
                "detectability model was calibrated over, so its recall estimate is an extrapolation."
            )
        return " ".join(parts)

    def _recommend(
        self,
        r: SufficiencyResult,
        specs: list[ObjectClassSpec],
        conditions: list[ObservationCondition],
    ) -> str:
        if r.level == SufficiencyLevel.SUFFICIENT:
            return "None: the evidence can settle this item."

        if r.coverage_required > 0.0 and r.coverage_achieved < r.coverage_required:
            missing = (r.coverage_required - r.coverage_achieved) * 100
            return (
                f"Extend coverage by about {missing:.0f} percentage points of the target extent. "
                f"Capture the unobserved portion rather than re-photographing what is already covered."
            )

        spec = next((s for s in specs if s.detector_label == r.limiting_class), specs[0] if specs else None)
        if spec is None:
            return "Capture additional imagery of this target."

        mean_gsd = float(np.mean([c.gsd_along_surface_m for c in conditions if c.gsd_along_surface_m > 0]) or 0)
        if mean_gsd > 0:
            achieved_px = spec.char_dim_installed_m / mean_gsd
            if achieved_px < spec.min_px_on_target:
                needed_range = spec.max_range_m()
                return (
                    f"Resolution-limited: '{spec.detector_label}' spans about {achieved_px:.0f} px "
                    f"against a budget of {spec.min_px_on_target} px. Re-photograph from within "
                    f"roughly {needed_range:.1f} m, or accept that this class cannot settle the item "
                    f"at the standoff the site allows."
                )
        if spec.self_occlusion_prior > 0.5:
            return (
                f"Occlusion-limited: '{spec.detector_label}' is hidden by its own assembly from "
                f"most viewpoints (self-occlusion prior {spec.self_occlusion_prior:.2f}). A different "
                f"viewing direction is needed, or this component must be verified by another means."
            )
        return "Capture additional imagery from a different direction, or escalate to physical inspection."

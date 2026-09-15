"""
decision.py — from evidence and sufficiency to an inspection state.

Only reached once sufficiency.py has said the evidence is capable of settling the question. The
rules here are deterministic and pattern-specific: what counts as settling a requirement depends
on what kind of thing the requirement asks for, and applying one rule to all of them is the
mistake that makes detection-driven compliance checking overclaim.

The clearest illustration is that a non-detection means opposite things under two of the patterns.
Under PresenceOfRequired, finding no barrier is evidence of non-compliance. Under VerifiedAbsence,
finding no worker inside an exclusion zone is evidence of compliance. The same detector output,
the same silence, opposite conclusions — and in both cases the conclusion is only licensed if the
detector would have spoken had there been something to speak about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from detectability import ObjectClassSpec, ObservationCondition, PrecisionCalibration
from sufficiency import (
    SufficiencyAssessor,
    SufficiencyLevel,
    SufficiencyModelKind,
    SufficiencyResult,
    union_coverage,
)


class ComplianceState(str, Enum):
    COMPLIANT = "Compliant"
    NON_COMPLIANT = "NonCompliant"
    INSUFFICIENT_EVIDENCE = "InsufficientEvidence"
    REQUIRES_HUMAN_VERIFICATION = "RequiresHumanVerification"
    NOT_APPLICABLE = "NotApplicable"
    OUT_OF_AUTOMATIC_SCOPE = "OutOfAutomaticScope"


class EvidencePattern(str, Enum):
    PRESENCE_OF_REQUIRED = "PresenceOfRequired"
    SPATIAL_COMPLETENESS = "SpatialCompleteness"
    COMPONENT_INTEGRITY = "ComponentIntegrity"
    VERIFIED_ABSENCE = "VerifiedAbsence"
    GEOMETRIC_CONFORMANCE = "GeometricConformance"
    DOCUMENTARY_CONFORMANCE = "DocumentaryConformance"
    MATERIAL_TEST_CONFORMANCE = "MaterialTestConformance"


@dataclass
class Detection:
    """A positive observation, already grounded to a target."""
    detection_id: str
    detector_label: str
    confidence: float
    image_id: str
    viewpoint_id: str = ""
    grounding_confidence: float = 1.0
    grounding_ambiguous: bool = False
    covered_sample_indices: list[int] = field(default_factory=list)

    def effective_confidence(self, precision: PrecisionCalibration) -> float:
        """
        Probability that this detection is a true positive of this class on this target.

        Two independent things must hold: the detector was right about what it saw, and the
        grounding was right about where it was. The first is the calibrated precision at this
        confidence, not the confidence itself.
        """
        return float(np.clip(precision.precision(self.confidence) * self.grounding_confidence, 0.0, 1.0))


@dataclass
class Requirement:
    requirement_id: str
    pattern: EvidencePattern
    required_classes: list[str] = field(default_factory=list)
    prohibited_classes: list[str] = field(default_factory=list)
    required_coverage: float | None = None
    clause_ref: str = ""
    regulation_id: str = ""
    checkability: str = "AUTOMATIC"
    statement: str = ""


@dataclass
class Decision:
    item_id: str
    requirement_id: str
    target_id: str
    state: ComplianceState
    rule_id: str
    rationale: str
    recommended_action: str = ""
    cites_clause: str = ""
    cites_regulation: str = ""
    sufficiency: SufficiencyResult | None = None
    supporting_detections: list[str] = field(default_factory=list)
    requires_human_verification: bool = False
    trace_complete: bool = False
    trace_broken_at: str = ""


class InspectionRuleEngine:
    """
    The deterministic rule layer. Ten rules, one per pattern plus the compound and scoping cases,
    each recorded with the id that appears in the decision so a result can be traced to the rule
    that produced it.
    """

    def __init__(
        self,
        assessor: SufficiencyAssessor,
        class_specs: dict[str, ObjectClassSpec],
        precision: PrecisionCalibration | None = None,
    ):
        self.assessor = assessor
        self.class_specs = class_specs
        self.precision = precision or assessor.precision

    # -- entry point -------------------------------------------------------

    def decide(
        self,
        item_id: str,
        target_id: str,
        requirement: Requirement,
        conditions: list[ObservationCondition],
        detections: list[Detection],
        prior_presence: float = 0.8,
        evidence_age_days: float = 0.0,
        applicable: bool = True,
        model_kind: SufficiencyModelKind = SufficiencyModelKind.BAYESIAN_ABSENCE,
    ) -> Decision:
        if not applicable:
            return Decision(
                item_id, requirement.requirement_id, target_id,
                ComplianceState.NOT_APPLICABLE, "R00",
                "Requirement is not scoped to this target in this phase.",
                cites_clause=requirement.clause_ref, cites_regulation=requirement.regulation_id,
                trace_complete=True,
            )

        if requirement.pattern in (
            EvidencePattern.DOCUMENTARY_CONFORMANCE,
            EvidencePattern.MATERIAL_TEST_CONFORMANCE,
        ):
            return self._rule_out_of_scope(item_id, target_id, requirement, "R09")

        if requirement.pattern == EvidencePattern.GEOMETRIC_CONFORMANCE:
            return self._rule_out_of_scope(item_id, target_id, requirement, "R10")

        # Any required class absent from the detector vocabulary makes the item unautomatable,
        # and saying so is different from saying the evidence was poor.
        unknown = [c for c in requirement.required_classes + requirement.prohibited_classes
                   if c not in self.class_specs]
        if unknown or (not requirement.required_classes and not requirement.prohibited_classes):
            d = self._rule_out_of_scope(item_id, target_id, requirement, "R09")
            d.rationale = (
                f"No detector class covers this requirement"
                + (f" (missing: {', '.join(unknown)})" if unknown else "")
                + ". Extend the detector vocabulary or record the item as manual."
            )
            return d

        required_specs = [self.class_specs[c] for c in requirement.required_classes]
        prohibited_specs = [self.class_specs[c] for c in requirement.prohibited_classes]

        # Compound requirements carry both a positive and a negative obligation; the item is only
        # as compliant as its weaker half.
        if required_specs and prohibited_specs and requirement.pattern != EvidencePattern.COMPONENT_INTEGRITY:
            return self._rule_compound(
                item_id, target_id, requirement, conditions, detections,
                required_specs, prohibited_specs, prior_presence, evidence_age_days, model_kind,
            )

        if requirement.pattern == EvidencePattern.PRESENCE_OF_REQUIRED:
            return self._rule_presence(item_id, target_id, requirement, conditions, detections,
                                       required_specs, prior_presence, evidence_age_days, model_kind)

        if requirement.pattern == EvidencePattern.SPATIAL_COMPLETENESS:
            return self._rule_completeness(item_id, target_id, requirement, conditions, detections,
                                           required_specs, prior_presence, evidence_age_days, model_kind)

        if requirement.pattern == EvidencePattern.COMPONENT_INTEGRITY:
            return self._rule_integrity(item_id, target_id, requirement, conditions, detections,
                                        required_specs, prohibited_specs, prior_presence,
                                        evidence_age_days, model_kind)

        if requirement.pattern == EvidencePattern.VERIFIED_ABSENCE:
            return self._rule_absence(item_id, target_id, requirement, conditions, detections,
                                      prohibited_specs or required_specs, prior_presence,
                                      evidence_age_days, model_kind)

        return self._rule_out_of_scope(item_id, target_id, requirement, "R09")

    # -- helpers -----------------------------------------------------------

    def _assess(self, item_id, specs, conditions, prior, age, model_kind, required_coverage=None,
                conjunctive=False) -> SufficiencyResult:
        if model_kind == SufficiencyModelKind.NAIVE_CLOSED_WORLD:
            return self.assessor.assess_naive(item_id, conditions)
        if model_kind == SufficiencyModelKind.HEURISTIC_SCORE:
            return self.assessor.assess_heuristic(item_id, conditions, detection_confidence=0.0,
                                                  evidence_age_days=age)
        return self.assessor.assess_bayesian(
            item_id, specs, conditions, prior_presence=prior, evidence_age_days=age,
            required_coverage=required_coverage, conjunctive=conjunctive,
        )

    def _positive(self, detections: list[Detection], labels: list[str]) -> list[Detection]:
        return [d for d in detections if d.detector_label in labels]

    def _best_positive(self, detections: list[Detection], labels: list[str]) -> Detection | None:
        hits = self._positive(detections, labels)
        if not hits:
            return None
        return max(hits, key=lambda d: d.effective_confidence(self.precision))

    def _finish(self, decision: Decision, requirement: Requirement) -> Decision:
        decision.cites_clause = requirement.clause_ref
        decision.cites_regulation = requirement.regulation_id
        decision.requires_human_verification = (
            decision.state == ComplianceState.REQUIRES_HUMAN_VERIFICATION
        )
        decision.trace_complete = bool(requirement.clause_ref and requirement.regulation_id)
        if not decision.trace_complete:
            decision.trace_broken_at = "requirement has no regulatory clause"
        return decision

    def _rule_out_of_scope(self, item_id, target_id, requirement, rule_id) -> Decision:
        return self._finish(Decision(
            item_id, requirement.requirement_id, target_id,
            ComplianceState.OUT_OF_AUTOMATIC_SCOPE, rule_id,
            f"Requirement is settled by {requirement.pattern.value}, which visual observation "
            f"cannot supply. Better photographs will never settle it; the correct route is the "
            f"document, test record or survey the clause actually asks for.",
            recommended_action="Route to the documentary or survey workflow.",
        ), requirement)

    # -- R01 presence of a required object ---------------------------------

    def _rule_presence(self, item_id, target_id, req, conditions, detections, specs,
                       prior, age, model_kind) -> Decision:
        suff = self._assess(item_id, specs, conditions, prior, age, model_kind)
        hit = self._best_positive(detections, req.required_classes)

        if hit is not None:
            conf = hit.effective_confidence(self.precision)
            if hit.grounding_ambiguous:
                return self._finish(Decision(
                    item_id, req.requirement_id, target_id,
                    ComplianceState.REQUIRES_HUMAN_VERIFICATION, "R01",
                    f"'{hit.detector_label}' detected at confidence {hit.confidence:.2f}, but its "
                    f"association with this target is ambiguous: more than one candidate target is "
                    f"compatible with the observation.",
                    recommended_action="Confirm which target the detected object belongs to.",
                    sufficiency=suff, supporting_detections=[hit.detection_id],
                ), req)
            if conf >= self.assessor.t.positive_confidence_high:
                return self._finish(Decision(
                    item_id, req.requirement_id, target_id,
                    ComplianceState.COMPLIANT, "R01",
                    f"'{hit.detector_label}' observed on this target. Calibrated precision at "
                    f"confidence {hit.confidence:.2f} is {self.precision.precision(hit.confidence):.2f}; "
                    f"combined with grounding confidence {hit.grounding_confidence:.2f} this gives "
                    f"{conf:.2f} that the required object is genuinely present here.",
                    sufficiency=suff, supporting_detections=[hit.detection_id],
                ), req)
            return self._finish(Decision(
                item_id, req.requirement_id, target_id,
                ComplianceState.REQUIRES_HUMAN_VERIFICATION, "R01",
                f"'{hit.detector_label}' detected, but combined detection and grounding confidence "
                f"is only {conf:.2f}.",
                recommended_action="Inspector to confirm the detection.",
                sufficiency=suff, supporting_detections=[hit.detection_id],
            ), req)

        # Nothing found. Whether that means anything depends entirely on sufficiency.
        return self._finish(self._from_absence(item_id, target_id, req, suff, "R01",
                                               settled_state=ComplianceState.NON_COMPLIANT), req)

    # -- R02 spatial completeness ------------------------------------------

    def _rule_completeness(self, item_id, target_id, req, conditions, detections, specs,
                           prior, age, model_kind) -> Decision:
        required_cov = req.required_coverage if req.required_coverage is not None \
            else self.assessor.t.required_coverage_default
        suff = self._assess(item_id, specs, conditions, prior, age, model_kind,
                            required_coverage=required_cov)

        observed_cov, exact = union_coverage([c for c in conditions if c.condition_valid])

        # The share of the extent that was both observed AND showed the required object.
        verified_cov, verified_exact = self._verified_coverage(
            conditions, detections, req.required_classes)

        # Continuity is a claim about which parts of the extent showed the object, and that
        # cannot be answered from a grounding that only says which zone an image belongs to.
        # Without per-sample association the system knows the barrier was seen somewhere and
        # the perimeter was covered somewhere, and has no way to tell whether those were the
        # same somewhere. Reporting a gap on that basis would be inventing one.
        if not verified_exact and self._positive(detections, req.required_classes):
            return self._finish(Decision(
                item_id, req.requirement_id, target_id,
                ComplianceState.REQUIRES_HUMAN_VERIFICATION, "R02",
                f"{observed_cov:.0%} of the target extent was observed and the required object was "
                f"detected, but the detections are grounded only at zone level, so there is no way "
                f"to tell which parts of the extent the object was seen on. Continuity cannot be "
                f"established or refuted from zone-level grounding, however good the imagery.",
                recommended_action=(
                    "Re-run the spatial grounding at detection level (level 2 or 3) so detections "
                    "carry the sample indices they cover, then re-evaluate."
                ),
                sufficiency=suff,
                supporting_detections=[d.detection_id for d in self._positive(detections, req.required_classes)],
            ), req)

        if model_kind != SufficiencyModelKind.NAIVE_CLOSED_WORLD and observed_cov < required_cov:
            state = (ComplianceState.REQUIRES_HUMAN_VERIFICATION
                     if observed_cov >= self.assessor.t.marginal_coverage
                     else ComplianceState.INSUFFICIENT_EVIDENCE)
            return self._finish(Decision(
                item_id, req.requirement_id, target_id, state, "R02",
                f"Continuity cannot be established: {observed_cov:.0%} of the target extent was "
                f"observed against a requirement of {required_cov:.0%}"
                + ("" if exact else " (coverage estimated without per-sample masks, so this is a lower bound)")
                + f". Of what was observed, {verified_cov:.0%} of the total extent showed the "
                f"required object. The unobserved remainder is not evidence either way.",
                recommended_action=suff.recommended_action,
                sufficiency=suff,
                supporting_detections=[d.detection_id for d in self._positive(detections, req.required_classes)],
            ), req)

        if verified_cov >= required_cov:
            return self._finish(Decision(
                item_id, req.requirement_id, target_id, ComplianceState.COMPLIANT, "R02",
                f"The required object was observed continuously over {verified_cov:.0%} of the "
                f"target extent, meeting the {required_cov:.0%} continuity requirement.",
                sufficiency=suff,
                supporting_detections=[d.detection_id for d in self._positive(detections, req.required_classes)],
            ), req)

        gap = observed_cov - verified_cov
        return self._finish(Decision(
            item_id, req.requirement_id, target_id, ComplianceState.NON_COMPLIANT, "R02",
            f"{observed_cov:.0%} of the extent was observed, which meets the coverage requirement, "
            f"but the required object was present over only {verified_cov:.0%}. A gap of about "
            f"{gap:.0%} of the extent was observed under conditions capable of showing the object "
            f"and did not show it.",
            recommended_action="Inspect the discontinuity on site.",
            sufficiency=suff,
            supporting_detections=[d.detection_id for d in self._positive(detections, req.required_classes)],
        ), req)

    def _verified_coverage(self, conditions, detections, labels) -> tuple[float, bool]:
        """
        Fraction of the whole extent both observed AND showing the required object.

        Returns (value, exact). Exact only when the observation conditions carry per-sample
        visibility masks and the detections carry the sample indices they cover, so that 'the
        barrier was seen here' and 'the perimeter was observed here' refer to the same physical
        points.

        When they do not, the honest return is (0.0, False), and the caller defers. There is no
        sound approximation available: multiplying coverage by detection confidence would mix a
        spatial quantity with an epistemic one and manufacture a gap wherever confidence is less
        than certainty, which is a fabricated finding rather than a conservative one.
        """
        valid = [c for c in conditions if c.condition_valid]
        if not valid:
            return 0.0, True

        hits = self._positive(detections, labels)
        if not hits:
            # Nothing was detected anywhere: verified coverage is genuinely zero, and that is
            # an exact answer rather than an unknown one.
            return 0.0, True

        n = valid[0].sampled_point_count
        masks = [c.sample_mask_array() for c in valid]
        if n > 0 and all(m is not None and len(m) == n for m in masks) \
                and any(h.covered_sample_indices for h in hits):
            observed = np.zeros(n, dtype=bool)
            for m in masks:
                observed |= m
            covered = np.zeros(n, dtype=bool)
            for h in hits:
                for i in h.covered_sample_indices:
                    if 0 <= i < n:
                        covered[i] = True
            return float((observed & covered).mean()), True

        return 0.0, False

    # -- R03 component integrity -------------------------------------------

    def _rule_integrity(self, item_id, target_id, req, conditions, detections, required_specs,
                        prohibited_specs, prior, age, model_kind) -> Decision:
        suff = self._assess(item_id, required_specs, conditions, prior, age, model_kind,
                            conjunctive=True)

        defect = self._best_positive(detections, req.prohibited_classes)
        if defect is not None and defect.effective_confidence(self.precision) >= self.assessor.t.positive_confidence_low:
            return self._finish(Decision(
                item_id, req.requirement_id, target_id, ComplianceState.NON_COMPLIANT, "R03",
                f"Defect class '{defect.detector_label}' observed on this assembly at confidence "
                f"{defect.confidence:.2f}. A positive defect detection settles the item directly; "
                f"no absence reasoning is needed.",
                recommended_action="Rectify and re-inspect.",
                sufficiency=suff, supporting_detections=[defect.detection_id],
            ), req)

        found = {d.detector_label for d in self._positive(detections, req.required_classes)
                 if d.effective_confidence(self.precision) >= self.assessor.t.positive_confidence_low}
        missing = [c for c in req.required_classes if c not in found]

        if not missing:
            return self._finish(Decision(
                item_id, req.requirement_id, target_id, ComplianceState.COMPLIANT, "R03",
                f"Every required component was observed: {', '.join(sorted(found))}.",
                sufficiency=suff,
                supporting_detections=[d.detection_id for d in self._positive(detections, req.required_classes)],
            ), req)

        # Something is missing from the report. Whether that is a finding depends on whether the
        # missing component could have been seen at all.
        limiting = suff.limiting_class or missing[0]
        spec = self.class_specs.get(limiting)
        extra = ""
        if spec is not None and spec.self_occlusion_prior > 0.5:
            extra = (
                f" '{limiting}' has a self-occlusion prior of {spec.self_occlusion_prior:.2f}: once "
                f"installed it is hidden by its own assembly from most viewpoints, so its absence "
                f"from the detector's report carries very little information about whether it is there."
            )
        d = self._from_absence(item_id, target_id, req, suff, "R03",
                               settled_state=ComplianceState.NON_COMPLIANT,
                               prefix=f"Components not observed: {', '.join(missing)}.{extra}")
        return self._finish(d, req)

    # -- R04 verified absence ----------------------------------------------

    def _rule_absence(self, item_id, target_id, req, conditions, detections, specs,
                      prior, age, model_kind) -> Decision:
        labels = req.prohibited_classes or req.required_classes
        # Prior here is the probability the prohibited thing is ABSENT, i.e. that the site is
        # compliant; the posterior machinery is the same, with presence and absence exchanged.
        suff = self._assess(item_id, specs, conditions, prior, age, model_kind)

        hit = self._best_positive(detections, labels)
        if hit is not None and hit.effective_confidence(self.precision) >= self.assessor.t.positive_confidence_low:
            return self._finish(Decision(
                item_id, req.requirement_id, target_id, ComplianceState.NON_COMPLIANT, "R04",
                f"'{hit.detector_label}' observed on this target at confidence {hit.confidence:.2f}, "
                f"where the requirement is that it be absent. A positive detection settles a "
                f"verified-absence requirement directly.",
                recommended_action="Clear the target and re-inspect.",
                sufficiency=suff, supporting_detections=[hit.detection_id],
            ), req)

        # Nothing detected. Under this pattern that is the compliant outcome — but only if the
        # looking was good enough that something would have been seen had it been there.
        return self._finish(self._from_absence(item_id, target_id, req, suff, "R04",
                                               settled_state=ComplianceState.COMPLIANT), req)

    # -- R05 compound ------------------------------------------------------

    def _rule_compound(self, item_id, target_id, req, conditions, detections,
                       required_specs, prohibited_specs, prior, age, model_kind) -> Decision:
        positive_req = Requirement(
            req.requirement_id, EvidencePattern.PRESENCE_OF_REQUIRED,
            required_classes=req.required_classes, required_coverage=req.required_coverage,
            clause_ref=req.clause_ref, regulation_id=req.regulation_id,
        )
        negative_req = Requirement(
            req.requirement_id, EvidencePattern.VERIFIED_ABSENCE,
            prohibited_classes=req.prohibited_classes,
            clause_ref=req.clause_ref, regulation_id=req.regulation_id,
        )
        a = self._rule_presence(item_id, target_id, positive_req, conditions, detections,
                                required_specs, prior, age, model_kind)
        b = self._rule_absence(item_id, target_id, negative_req, conditions, detections,
                               prohibited_specs, prior, age, model_kind)

        # The compound is only as good as its weaker half, in this order of severity.
        order = [
            ComplianceState.NON_COMPLIANT,
            ComplianceState.INSUFFICIENT_EVIDENCE,
            ComplianceState.REQUIRES_HUMAN_VERIFICATION,
            ComplianceState.OUT_OF_AUTOMATIC_SCOPE,
            ComplianceState.COMPLIANT,
        ]
        worse = a if order.index(a.state) <= order.index(b.state) else b
        return self._finish(Decision(
            item_id, req.requirement_id, target_id, worse.state, "R05",
            f"Compound requirement, evaluated as two obligations and resolved to the weaker. "
            f"Positive obligation ({', '.join(req.required_classes)}): {a.state.value}. "
            f"Negative obligation ({', '.join(req.prohibited_classes)}): {b.state.value}. "
            f"Governing half: {worse.rationale}",
            recommended_action=worse.recommended_action,
            sufficiency=worse.sufficiency,
            supporting_detections=a.supporting_detections + b.supporting_detections,
        ), req)

    # -- shared absence handling -------------------------------------------

    def _from_absence(self, item_id, target_id, req, suff: SufficiencyResult, rule_id: str,
                      settled_state: ComplianceState, prefix: str = "") -> Decision:
        """
        Turn a non-detection into a state, according to how much the non-detection is worth.

        `settled_state` is what a well-evidenced silence means under this pattern: non-compliance
        when something required was not found, compliance when something prohibited was not found.
        """
        head = (prefix + " ") if prefix else ""

        if suff.level == SufficiencyLevel.SUFFICIENT:
            return Decision(
                item_id, req.requirement_id, target_id, settled_state, rule_id,
                f"{head}Nothing was detected, and the evidence was capable of detecting it: "
                f"{suff.rationale}",
                recommended_action=(
                    "Rectify and re-inspect." if settled_state == ComplianceState.NON_COMPLIANT
                    else "None: absence is verified."
                ),
                sufficiency=suff,
            )

        if suff.level == SufficiencyLevel.MARGINAL:
            return Decision(
                item_id, req.requirement_id, target_id,
                ComplianceState.REQUIRES_HUMAN_VERIFICATION, rule_id,
                f"{head}Nothing was detected, and the evidence is marginal: {suff.rationale}",
                recommended_action=suff.recommended_action or "Inspector review required.",
                sufficiency=suff,
            )

        return Decision(
            item_id, req.requirement_id, target_id,
            ComplianceState.INSUFFICIENT_EVIDENCE, rule_id,
            f"{head}Nothing was detected, but the evidence could not have detected it either: "
            f"{suff.rationale} A non-detection under these conditions is not evidence of absence.",
            recommended_action=suff.recommended_action,
            sufficiency=suff,
        )

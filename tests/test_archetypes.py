"""
test_archetypes.py — does the reasoning behave correctly on the four evidential archetypes?

These are behaviour tests, not unit tests. Each one sets up an observation geometry and asks
whether the framework reaches the state a competent inspector would reach given the same
information — and, just as importantly, declines where a competent inspector would decline.

The case that matters most is test_washer_is_unobservable. A bolt and a nut are detected; a washer
is not. Naive closed-world reasoning calls that a missing washer and reports non-compliance. The
correct answer is that the washer was never observable in the first place, because once the nut is
torqued down only a 4 mm edge annulus shows, and at any realistic site standoff that is well below
the detector's pixel budget. If the framework gets this one right it has earned its complexity.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detectability import (  # noqa: E402
    DetectabilityModel,
    ObservationCondition,
    PrecisionCalibration,
    load_object_classes,
)
from decision import (  # noqa: E402
    ComplianceState,
    Decision,
    Detection,
    EvidencePattern,
    InspectionRuleEngine,
    Requirement,
)
from sufficiency import (  # noqa: E402
    SufficiencyAssessor,
    SufficiencyLevel,
    SufficiencyModelKind,
    SufficiencyThresholds,
    posterior_absence,
    posterior_shift,
    union_coverage,
)

DATA = Path(__file__).resolve().parents[1] / "data"
CLASSES = load_object_classes(DATA / "object_classes.csv")

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"\n        {detail}" if detail else ""))


def mask_for(n_samples: int, visible_indices) -> str:
    """Build the hex bitmask the Unity analyser emits, for constructing test fixtures."""
    value = 0
    for i in visible_indices:
        value |= 1 << i
    return format(value, "x")


def make_condition(
    target_id: str,
    viewpoint_id: str,
    *,
    coverage: float,
    distance_m: float,
    incidence_deg: float = 15.0,
    unoccluded: float = 0.95,
    extent: float = 40.0,
    n_samples: int = 64,
    hfov_deg: float = 60.0,
    width_px: int = 4000,
    visible_indices=None,
) -> ObservationCondition:
    gsd_perp = 2 * distance_m * math.tan(math.radians(hfov_deg) / 2) / width_px
    gsd_along = gsd_perp / max(math.cos(math.radians(incidence_deg)), 0.05)

    if visible_indices is None:
        visible_indices = range(int(round(coverage * n_samples)))
    visible_indices = list(visible_indices)

    return ObservationCondition(
        target_id=target_id,
        viewpoint_id=viewpoint_id,
        image_id=f"IMG_{viewpoint_id}",
        target_extent=extent,
        extent_unit="m",
        frustum_coverage=min(coverage / max(unoccluded, 1e-6), 1.0),
        unoccluded_fraction=unoccluded,
        effective_coverage=coverage,
        viewing_distance_m=distance_m,
        incidence_angle_deg=incidence_deg,
        gsd_perpendicular_m=gsd_perp,
        gsd_along_surface_m=gsd_along,
        sampled_point_count=n_samples,
        visible_point_count=len(visible_indices),
        pose_uncertainty_m=0.2,
        condition_valid=True,
        visible_sample_mask=mask_for(n_samples, visible_indices),
    )


def engine(thresholds: SufficiencyThresholds | None = None) -> InspectionRuleEngine:
    """An UNCALIBRATED engine: the state the framework is in before any calibration has run."""
    model = DetectabilityModel()
    precision = PrecisionCalibration()
    assessor = SufficiencyAssessor(model, thresholds or SufficiencyThresholds(), precision)
    return InspectionRuleEngine(assessor, CLASSES, precision)


def calibrated_engine(thresholds: SufficiencyThresholds | None = None) -> InspectionRuleEngine:
    """
    An engine standing in for one whose detectability model has been fitted on held-out data.

    The slope intervals here are plausible values, not measured ones: this fixture exists to
    exercise the code path a calibrated model takes, not to substitute for calibration. Every
    figure a paper reports must come from `DetectabilityModel.fit` on the project's own data.
    """
    model = DetectabilityModel(
        b_resolution=0.92, b_occlusion=-3.85, b_incidence=-2.10,
        calibrated=True, n_calibration=1240,
        slope_ci={"b_resolution": (0.78, 1.06),
                  "b_occlusion": (-4.40, -3.30),
                  "b_incidence": (-2.60, -1.60)},
        px_range=(4.0, 900.0), incidence_range=(0.0, 85.0),
    )
    precision = PrecisionCalibration()
    assessor = SufficiencyAssessor(model, thresholds or SufficiencyThresholds(), precision)
    return InspectionRuleEngine(assessor, CLASSES, precision)


# ---------------------------------------------------------------------------
# Archetype A — presence of a required object (excavation edge protection)
# ---------------------------------------------------------------------------

def test_edge_protection():
    print("\nArchetype A — excavation edge protection (IR-P3-008, PresenceOfRequired)")
    eng = calibrated_engine()
    req = Requirement(
        "IR-P3-008", EvidencePattern.PRESENCE_OF_REQUIRED,
        required_classes=["barrier", "fence", "barricade"],
        clause_ref="Art. 118 comma 3", regulation_id="REG-DLGS-81-2008",
    )

    good = [make_condition("ExcavationBoundary_01", f"VP{i}", coverage=0.95, distance_m=12.0)
            for i in range(6)]

    d = eng.decide("ITEM-A1", "ExcavationBoundary_01", req, good,
                   [Detection("DET-1", "barrier", 0.93, "IMG_VP0", grounding_confidence=0.97)])
    check("barrier detected under good conditions -> Compliant",
          d.state == ComplianceState.COMPLIANT, d.rationale[:150])

    d = eng.decide("ITEM-A2", "ExcavationBoundary_01", req, good, [])
    check("no barrier, good conditions -> NonCompliant",
          d.state == ComplianceState.NON_COMPLIANT,
          f"posterior absence {d.sufficiency.posterior_absence:.3f}, "
          f"aggregate recall {d.sufficiency.aggregated_recall:.3f}")

    poor = [make_condition("ExcavationBoundary_01", "VP9", coverage=0.10,
                           distance_m=45.0, incidence_deg=78.0, unoccluded=0.35)]
    d = eng.decide("ITEM-A3", "ExcavationBoundary_01", req, poor, [])
    check("no barrier, poor conditions -> InsufficientEvidence",
          d.state == ComplianceState.INSUFFICIENT_EVIDENCE,
          f"posterior absence {d.sufficiency.posterior_absence:.3f} "
          f"(prior absent 0.20), aggregate recall {d.sufficiency.aggregated_recall:.3f}")

    d = eng.decide("ITEM-A4", "ExcavationBoundary_01", req, poor, [],
                   model_kind=SufficiencyModelKind.NAIVE_CLOSED_WORLD)
    check("same poor evidence under closed-world -> NonCompliant (the error being measured)",
          d.state == ComplianceState.NON_COMPLIANT)

    # A marginal detection is confirmed by a human rather than accepted outright.
    d = eng.decide("ITEM-A5", "ExcavationBoundary_01", req, good,
                   [Detection("DET-2", "barrier", 0.72, "IMG_VP0", grounding_confidence=0.80)])
    check("a marginal detection -> RequiresHumanVerification, not Compliant",
          d.state == ComplianceState.REQUIRES_HUMAN_VERIFICATION, d.rationale[:130])

    d = eng.decide("ITEM-A6", "ExcavationBoundary_01", req, good,
                   [Detection("DET-3", "barrier", 0.93, "IMG_VP0",
                              grounding_confidence=0.55, grounding_ambiguous=True)])
    check("an ambiguously grounded detection -> RequiresHumanVerification",
          d.state == ComplianceState.REQUIRES_HUMAN_VERIFICATION)


def test_calibration_is_a_precondition_for_accusation():
    """
    The framework will not assert non-compliance on an uncalibrated detectability model.

    This is a designed property, not an accident of the defaults. Asserting that something
    required is missing rests entirely on the claim 'we would have seen it', and that claim is
    only as good as the calibration behind it. With no calibration the model applies a blanket
    haircut to recall, the aggregate never reaches the bar the posterior threshold implies, and
    the system defers instead of accusing.
    """
    print("\nCalibration is a precondition for asserting non-compliance")
    req = Requirement(
        "IR-P3-008", EvidencePattern.PRESENCE_OF_REQUIRED,
        required_classes=["barrier"],
        clause_ref="Art. 118 comma 3", regulation_id="REG-DLGS-81-2008",
    )
    good = [make_condition("ExcavationBoundary_01", f"VP{i}", coverage=0.95, distance_m=12.0)
            for i in range(6)]

    un = engine().decide("ITEM-CAL1", "ExcavationBoundary_01", req, good, [])
    cal = calibrated_engine().decide("ITEM-CAL2", "ExcavationBoundary_01", req, good, [])

    check("uncalibrated model defers rather than accusing",
          un.state != ComplianceState.NON_COMPLIANT,
          f"uncalibrated: {un.state.value}, aggregate recall {un.sufficiency.aggregated_recall:.3f}, "
          f"posterior {un.sufficiency.posterior_absence:.3f}")
    check("calibrated model on identical evidence settles it",
          cal.state == ComplianceState.NON_COMPLIANT,
          f"calibrated: {cal.state.value}, aggregate recall {cal.sufficiency.aggregated_recall:.3f}, "
          f"posterior {cal.sufficiency.posterior_absence:.3f}")


# ---------------------------------------------------------------------------
# Archetype B — spatial completeness (guardrail continuity)
# ---------------------------------------------------------------------------

def test_guardrail_continuity():
    print("\nArchetype B — guardrail continuity (IR-P6-032, SpatialCompleteness)")
    eng = calibrated_engine()
    req = Requirement(
        "IR-P6-032", EvidencePattern.SPATIAL_COMPLETENESS,
        required_classes=["barrier", "traffic barrier"],
        required_coverage=0.95,
        clause_ref="Road restraint systems", regulation_id="REG-UNI-EN-1317",
    )

    # One confident look at one third of the alignment.
    partial = [make_condition("RoadRestraint_01", "VP0", coverage=0.33, distance_m=10.0,
                              extent=120.0, visible_indices=range(0, 21))]
    d = eng.decide("ITEM-B1", "RoadRestraint_01", req, partial,
                   [Detection("DET-10", "barrier", 0.93, "IMG_VP0",
                              grounding_confidence=0.97, covered_sample_indices=list(range(0, 21)))])
    check("confident detection over one third of the alignment -> not Compliant",
          d.state != ComplianceState.COMPLIANT,
          f"observed coverage {d.sufficiency.coverage_achieved:.2f}; state {d.state.value}")

    # Three overlapping passes covering the whole run.
    full = [
        make_condition("RoadRestraint_01", "VP0", coverage=0.45, distance_m=10.0,
                       extent=120.0, visible_indices=range(0, 29)),
        make_condition("RoadRestraint_01", "VP1", coverage=0.45, distance_m=10.0,
                       extent=120.0, visible_indices=range(24, 53)),
        make_condition("RoadRestraint_01", "VP2", coverage=0.30, distance_m=10.0,
                       extent=120.0, visible_indices=range(48, 64)),
    ]
    covered_all = list(range(0, 64))
    d = eng.decide("ITEM-B2", "RoadRestraint_01", req, full, [
        Detection("DET-11", "barrier", 0.91, "IMG_VP0", grounding_confidence=0.96,
                  covered_sample_indices=covered_all),
    ])
    check("three overlapping passes covering the whole run -> Compliant",
          d.state == ComplianceState.COMPLIANT,
          f"observed coverage {d.sufficiency.coverage_achieved:.2f}")

    # Whole run observed, but the barrier is missing over the middle third.
    d = eng.decide("ITEM-B3", "RoadRestraint_01", req, full, [
        Detection("DET-12", "barrier", 0.91, "IMG_VP0", grounding_confidence=0.96,
                  covered_sample_indices=list(range(0, 22)) + list(range(43, 64))),
    ])
    check("whole run observed but a gap in the middle -> NonCompliant",
          d.state == ComplianceState.NON_COMPLIANT, d.rationale[:170])

    cov, exact = union_coverage(full)
    check("per-sample masks give an exact union rather than a bound",
          exact and abs(cov - 1.0) < 1e-9, f"union coverage {cov:.3f}, exact={exact}")


# ---------------------------------------------------------------------------
# Archetype C — component integrity (the washer)
# ---------------------------------------------------------------------------

def test_washer_is_unobservable():
    print("\nArchetype C — bolted connection (IR-P4-019, ComponentIntegrity)")
    eng = calibrated_engine()
    req = Requirement(
        "IR-P4-019", EvidencePattern.COMPONENT_INTEGRITY,
        required_classes=["bolt", "nut", "washer"],
        prohibited_classes=["loose bolt", "missing bolt"],
        clause_ref="Inspection documents", regulation_id="REG-EN-ISO-16228",
    )

    # A close, well-exposed look at the connection: 2 m, near face-on.
    close = [make_condition("BoltedConnection_07", f"VP{i}", coverage=0.95, distance_m=2.0,
                            incidence_deg=10.0, extent=12, n_samples=12) for i in range(3)]

    d = eng.decide("ITEM-C1", "BoltedConnection_07", req, close, [
        Detection("DET-20", "bolt", 0.88, "IMG_VP0", grounding_confidence=0.95),
        Detection("DET-21", "nut", 0.86, "IMG_VP0", grounding_confidence=0.95),
    ])
    check("bolt and nut seen, washer not -> InsufficientEvidence, not NonCompliant",
          d.state == ComplianceState.INSUFFICIENT_EVIDENCE,
          f"limiting class '{d.sufficiency.limiting_class}', "
          f"aggregate recall {d.sufficiency.aggregated_recall:.4f}, "
          f"posterior absence {d.sufficiency.posterior_absence:.3f}")

    check("the washer is named as the limiting factor",
          d.sufficiency.limiting_class == "washer")
    check("the recommendation is actionable, not generic",
          "washer" in d.recommended_action.lower(), d.recommended_action[:170])

    d = eng.decide("ITEM-C2", "BoltedConnection_07", req, close, [
        Detection("DET-22", "bolt", 0.88, "IMG_VP0", grounding_confidence=0.95),
        Detection("DET-23", "nut", 0.86, "IMG_VP0", grounding_confidence=0.95),
        Detection("DET-24", "washer", 0.71, "IMG_VP0", grounding_confidence=0.95),
    ])
    check("all three components seen -> Compliant", d.state == ComplianceState.COMPLIANT)

    d = eng.decide("ITEM-C3", "BoltedConnection_07", req, close, [
        Detection("DET-25", "bolt", 0.88, "IMG_VP0", grounding_confidence=0.95),
        Detection("DET-26", "missing bolt", 0.79, "IMG_VP0", grounding_confidence=0.93),
    ])
    check("a positive defect detection settles the item directly -> NonCompliant",
          d.state == ComplianceState.NON_COMPLIANT, d.rationale[:130])

    d = eng.decide("ITEM-C4", "BoltedConnection_07", req, close, [
        Detection("DET-27", "bolt", 0.88, "IMG_VP0", grounding_confidence=0.95),
        Detection("DET-28", "nut", 0.86, "IMG_VP0", grounding_confidence=0.95),
    ], model_kind=SufficiencyModelKind.NAIVE_CLOSED_WORLD)
    check("same evidence under closed-world -> NonCompliant (the false accusation)",
          d.state == ComplianceState.NON_COMPLIANT)


# ---------------------------------------------------------------------------
# Archetype D — verified absence (site restoration)
# ---------------------------------------------------------------------------

def test_site_restoration():
    print("\nArchetype D — removal of temporary works (IR-P6-034, VerifiedAbsence)")
    eng = calibrated_engine()
    req = Requirement(
        "IR-P6-034", EvidencePattern.VERIFIED_ABSENCE,
        prohibited_classes=["barricade", "container", "fence", "traffic barrier"],
        clause_ref="Titolo IV Capo I", regulation_id="REG-DLGS-81-2008",
    )

    thorough = [make_condition("SiteArea_01", f"VP{i}", coverage=0.92, distance_m=15.0,
                               incidence_deg=20.0, extent=2400.0) for i in range(8)]
    d = eng.decide("ITEM-D1", "SiteArea_01", req, thorough, [], prior_presence=0.75)
    check("thorough sweep finds nothing -> Compliant (absence verified)",
          d.state == ComplianceState.COMPLIANT,
          f"aggregate recall {d.sufficiency.aggregated_recall:.3f}, "
          f"posterior {d.sufficiency.posterior_absence:.3f}")

    d = eng.decide("ITEM-D2", "SiteArea_01", req, thorough,
                   [Detection("DET-30", "container", 0.90, "IMG_VP3", grounding_confidence=0.94)],
                   prior_presence=0.75)
    check("a container is still standing -> NonCompliant",
          d.state == ComplianceState.NON_COMPLIANT)

    sparse = [make_condition("SiteArea_01", "VP0", coverage=0.18, distance_m=60.0,
                             incidence_deg=70.0, unoccluded=0.5, extent=2400.0)]
    d = eng.decide("ITEM-D3", "SiteArea_01", req, sparse, [], prior_presence=0.75)
    check("one distant oblique frame finds nothing -> not Compliant",
          d.state != ComplianceState.COMPLIANT,
          f"state {d.state.value}, aggregate recall {d.sufficiency.aggregated_recall:.3f}")


# ---------------------------------------------------------------------------
# PPE, and the properties of the Bayesian core
# ---------------------------------------------------------------------------

def test_ppe():
    print("\nPPE (IR-P1-003, ComponentIntegrity) — helmet, vest and boots differ in observability")
    eng = calibrated_engine()
    req = Requirement(
        "IR-P1-003", EvidencePattern.COMPONENT_INTEGRITY,
        required_classes=["helmet", "vest", "boots"],
        clause_ref="Titolo III Capo II Art. 74-79", regulation_id="REG-DLGS-81-2008",
    )
    conds = [make_condition("Worker_12", f"VP{i}", coverage=0.9, distance_m=8.0,
                            incidence_deg=15.0, extent=1, n_samples=8) for i in range(2)]
    d = eng.decide("ITEM-E1", "Worker_12", req, conds, [
        Detection("DET-40", "helmet", 0.91, "IMG_VP0", grounding_confidence=0.96),
        Detection("DET-41", "vest", 0.88, "IMG_VP0", grounding_confidence=0.96),
    ])
    check("helmet and vest seen, boots not -> not NonCompliant",
          d.state != ComplianceState.NON_COMPLIANT,
          f"state {d.state.value}, limiting class '{d.sufficiency.limiting_class}'")
    check("boots are identified as the limiting component",
          d.sufficiency.limiting_class == "boots",
          "feet are occluded by terrain, spoil and other workers in most site imagery")


def test_bayesian_properties():
    print("\nProperties of the Bayesian absence core")

    check("perfect recall drives the posterior to certainty of absence",
          posterior_absence(1.0, 0.8) > 0.999,
          f"P = {posterior_absence(1.0, 0.8):.6f}")

    p0 = posterior_absence(0.0, 0.8)
    check("zero recall leaves the posterior at the prior",
          abs(p0 - 0.2) < 1e-9, f"P = {p0:.6f}, prior absent = 0.20")

    check("zero recall means zero information gain",
          abs(posterior_shift(p0, 0.8)) < 1e-9)

    monotone = all(
        posterior_absence(d, 0.8) < posterior_absence(d + 0.05, 0.8)
        for d in [x / 100 for x in range(0, 95, 5)]
    )
    check("the posterior is strictly increasing in aggregate recall", monotone)

    # The threshold implies a demanding evidentiary bar, by design.
    need = min(d / 1000 for d in range(1000) if posterior_absence(d / 1000, 0.8) >= 0.90)
    check("asserting non-compliance at prior 0.8 requires aggregate recall above 0.95",
          need > 0.95, f"minimum aggregate recall to reach P>=0.90 is {need:.3f}")

    strict = posterior_absence(0.9, 0.95)
    lax = posterior_absence(0.9, 0.5)
    check("a stronger prior of compliance makes non-compliance harder to assert",
          strict < lax, f"prior 0.95 -> {strict:.3f}; prior 0.50 -> {lax:.3f}")


def test_out_of_scope():
    print("\nRequirements outside automatic scope are distinguished from poor evidence")
    eng = engine()
    doc = Requirement("IR-P3-011", EvidencePattern.DOCUMENTARY_CONFORMANCE,
                      clause_ref="Conformity control", regulation_id="REG-UNI-7163-72")
    d = eng.decide("ITEM-F1", "ConcreteBatch_04", doc, [], [])
    check("a documentary requirement -> OutOfAutomaticScope",
          d.state == ComplianceState.OUT_OF_AUTOMATIC_SCOPE)

    gap = Requirement("IR-P2-006", EvidencePattern.PRESENCE_OF_REQUIRED,
                      required_classes=["drilling rig"],
                      clause_ref="Titolo IV Capo I", regulation_id="REG-DLGS-81-2008")
    d = eng.decide("ITEM-F2", "DrillingRig_01", gap, [], [])
    check("a class absent from the detector vocabulary -> OutOfAutomaticScope",
          d.state == ComplianceState.OUT_OF_AUTOMATIC_SCOPE, d.rationale[:130])


def test_traceability():
    print("\nTraceability")
    eng = calibrated_engine()
    req = Requirement("IR-P3-008", EvidencePattern.PRESENCE_OF_REQUIRED,
                      required_classes=["barrier"],
                      clause_ref="Art. 118 comma 3", regulation_id="REG-DLGS-81-2008")
    conds = [make_condition("ExcavationBoundary_01", "VP0", coverage=0.8, distance_m=12.0)]
    d = eng.decide("ITEM-G1", "ExcavationBoundary_01", req, conds,
                   [Detection("DET-50", "barrier", 0.84, "IMG_VP0", grounding_confidence=0.95)])
    check("a decision cites its clause and regulation",
          d.trace_complete and d.cites_clause and d.cites_regulation,
          f"{d.cites_regulation} / {d.cites_clause}")

    ungrounded = Requirement("IR-XXX", EvidencePattern.PRESENCE_OF_REQUIRED,
                             required_classes=["barrier"])
    d = eng.decide("ITEM-G2", "ExcavationBoundary_01", ungrounded, conds, [])
    check("a requirement with no clause yields an incomplete trace",
          not d.trace_complete and d.trace_broken_at, d.trace_broken_at)


def test_ablation_divergence():
    print("\nAblation — where the models disagree")
    eng = calibrated_engine()
    req = Requirement("IR-P3-008", EvidencePattern.PRESENCE_OF_REQUIRED,
                      required_classes=["barrier"],
                      clause_ref="Art. 118", regulation_id="REG-DLGS-81-2008")

    rows = []
    for label, conds in [
        ("good  (4 views, 12 m, coverage 0.80)",
         [make_condition("B", f"V{i}", coverage=0.80, distance_m=12.0) for i in range(4)]),
        ("fair  (2 views, 25 m, coverage 0.45)",
         [make_condition("B", f"V{i}", coverage=0.45, distance_m=25.0, incidence_deg=45.0) for i in range(2)]),
        ("poor  (1 view, 45 m, coverage 0.10)",
         [make_condition("B", "V0", coverage=0.10, distance_m=45.0, incidence_deg=78.0, unoccluded=0.35)]),
    ]:
        states = {}
        for kind in (SufficiencyModelKind.NAIVE_CLOSED_WORLD,
                     SufficiencyModelKind.HEURISTIC_SCORE,
                     SufficiencyModelKind.BAYESIAN_ABSENCE):
            states[kind] = eng.decide(f"AB-{label}", "B", req, conds, [], model_kind=kind).state
        rows.append((label, states))
        print(f"        {label:38s} "
              f"naive={states[SufficiencyModelKind.NAIVE_CLOSED_WORLD].value:22s} "
              f"heuristic={states[SufficiencyModelKind.HEURISTIC_SCORE].value:26s} "
              f"bayesian={states[SufficiencyModelKind.BAYESIAN_ABSENCE].value}")

    check("closed-world settles every case regardless of evidence quality",
          all(s[SufficiencyModelKind.NAIVE_CLOSED_WORLD] == ComplianceState.NON_COMPLIANT
              for _, s in rows))
    check("the Bayesian model declines on the poor case",
          rows[-1][1][SufficiencyModelKind.BAYESIAN_ABSENCE] == ComplianceState.INSUFFICIENT_EVIDENCE)
    check("the models diverge, so the ablation measures something",
          len({tuple(sorted(s.value for s in st.values())) for _, st in rows}) > 1)


def test_class_ranges():
    print("\nMaximum range at which each class can be resolved (60 deg HFOV, 4000 px)")
    for label in ["helmet", "vest", "boots", "barrier", "bolt", "nut", "washer", "steel rebar"]:
        spec = CLASSES[label]
        print(f"        {label:14s} char dim {spec.char_dim_installed_m*1000:7.1f} mm  "
              f"-> max range {spec.max_range_m():7.2f} m")
    check("the washer's resolvable range is under a metre",
          CLASSES["washer"].max_range_m() < 1.0,
          f"{CLASSES['washer'].max_range_m():.2f} m: this is why IR-P4-019 cannot be settled "
          f"from general site photography")
    check("a barrier is resolvable across a whole site",
          CLASSES["barrier"].max_range_m() > 100.0,
          f"{CLASSES['barrier'].max_range_m():.1f} m")


if __name__ == "__main__":
    for fn in (test_edge_protection, test_guardrail_continuity, test_washer_is_unobservable,
               test_site_restoration, test_ppe, test_calibration_is_a_precondition_for_accusation,
               test_bayesian_properties, test_out_of_scope,
               test_traceability, test_ablation_divergence, test_class_ranges):
        fn()

    print(f"\n{'='*78}\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for f in FAILED:
            print(f"  FAILED: {f}")
        sys.exit(1)

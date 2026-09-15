using System;
using System.Collections.Generic;
using UnityEngine;

namespace Cieo.Visibility
{
    /// <summary>
    /// A camera pose recovered from the reconstruction pipeline, expressed in project coordinates.
    /// </summary>
    [Serializable]
    public class Viewpoint
    {
        public string viewpointId;
        public string imageId;
        public string capturedAt;          // ISO 8601

        public Vector3 position;
        public Quaternion rotation;

        public float horizontalFovDeg = 60f;
        public int imageWidthPx = 4000;
        public int imageHeightPx = 3000;

        /// <summary>"surveyed" | "estimated" | "assumed". Poses from a depth-estimation and
        /// registration pipeline are "estimated": metrically scaled, but not survey grade.</summary>
        public string poseSource = "estimated";

        /// <summary>1-sigma position uncertainty in metres. Propagates into grounding confidence
        /// and, through it, into the sufficiency of every decision this viewpoint supports.</summary>
        public float poseUncertaintyM = 0.25f;
    }

    /// <summary>
    /// The observation-condition vector for one target seen from one viewpoint: the geometry of a
    /// single act of looking. This is what the detectability model is conditioned on, and it is the
    /// quantity the image itself cannot supply.
    /// </summary>
    [Serializable]
    public class ObservationConditionRecord
    {
        public string targetId;
        public string targetClass;
        public string viewpointId;
        public string imageId;
        public string capturedAt;

        public float targetExtent;
        public string extentUnit;

        public float frustumCoverage;      // share of extent inside the frustum
        public float unoccludedFraction;   // of the in-frustum share, the part with clear line of sight
        public float effectiveCoverage;    // product of the two: the share actually visible

        public float viewingDistanceM;     // mean over visible samples
        public float nearestDistanceM;
        public float incidenceAngleDeg;    // mean over visible samples, 0 = face-on
        public float gsdPerpendicularM;    // metres per pixel at the mean range
        public float gsdAlongSurfaceM;     // corrected for foreshortening at the mean incidence

        public int sampledPointCount;
        public int visiblePointCount;
        public float poseUncertaintyM;

        /// <summary>
        /// Per-sample visibility, as a little-endian hex bitmask over the target's sample set:
        /// bit i is set when sample i was visible from this viewpoint.
        ///
        /// This is what makes exact coverage unions possible. Because a target's sampling is
        /// deterministic in its seed, sample index i denotes the same physical point in every
        /// viewpoint's record, so the coverage of two viewpoints combines by bitwise OR rather
        /// than by an assumption about how their coverage fractions overlap. Assuming they are
        /// disjoint is how a system convinces itself it has covered a whole perimeter it has
        /// only seen one end of, twice.
        /// </summary>
        public string visibleSampleMask = "";

        public bool conditionValid = true;
        public string invalidReason = "";
    }

    /// <summary>
    /// Computes, for a target and a viewpoint, how much of the target could actually have been
    /// seen — by frustum test, back-face test and ray cast against scene geometry.
    ///
    /// The point of doing this in the virtual environment rather than on the image is that the
    /// scene holds two things the image does not: the target's full extent, including the part
    /// that fell outside the frame, and everything standing between the target and the camera.
    /// Without both, a detector's silence cannot be interpreted.
    ///
    /// Occluder set
    /// ------------
    /// The result depends entirely on what is treated as an occluder. A run against design BIM
    /// geometry alone will overstate visibility, because the spoil heaps, plant, stored materials
    /// and site huts that actually block the view are not in the design model. Include the
    /// registered as-built point cloud mesh in the occluder mask wherever it exists, and record
    /// in <see cref="occluderSetDescription"/> what was included, because a coverage figure is
    /// only interpretable alongside it.
    /// </summary>
    public class VisibilityAnalyzer : MonoBehaviour
    {
        [Header("Occlusion")]
        [Tooltip("Layers treated as occluders. Include the as-built reconstruction, not only the " +
                 "design model, or coverage will be optimistic.")]
        public LayerMask occluderMask = ~0;

        [Tooltip("Recorded into the output and into the PROV activity for this analysis.")]
        public string occluderSetDescription = "design BIM + registered as-built mesh";

        [Tooltip("Ray origins are lifted off the surface by this much to avoid self-intersection " +
                 "at the sample point itself.")]
        public float rayOriginEpsilon = 0.01f;

        [Tooltip("Samples whose normal faces away from the camera are counted as occluded rather " +
                 "than skipped. Correct for solid targets: the far side of a barrier is genuinely " +
                 "not observable from this viewpoint.")]
        public bool backFaceCounts = true;

        [Header("Provenance")]
        public string analysisEngine = "Unity";
        public string analysisVersion = "cieo-visibility/0.1.0";
        public string geometrySource = "design BIM + as-built reconstruction";

        /// <summary>
        /// Analyse one target from one viewpoint.
        /// </summary>
        public ObservationConditionRecord Analyse(InspectionTargetMarker target, Viewpoint vp)
        {
            var rec = new ObservationConditionRecord
            {
                targetId = target.targetLocalName,
                targetClass = target.cieoClass,
                viewpointId = vp.viewpointId,
                imageId = vp.imageId,
                capturedAt = vp.capturedAt,
                extentUnit = target.ExtentUnit(),
                targetExtent = target.ComputeExtent(),
                poseUncertaintyM = vp.poseUncertaintyM
            };

            if (target.PolylineIsDerived)
            {
                rec.conditionValid = false;
                rec.invalidReason = "linear extent was derived from renderer bounds rather than an " +
                                    "authored polyline; coverage over a guessed extent is not meaningful";
            }

            var samples = target.Sample();
            rec.sampledPointCount = samples.Count;
            if (samples.Count == 0)
            {
                rec.conditionValid = false;
                rec.invalidReason = string.IsNullOrEmpty(rec.invalidReason)
                    ? "target produced no surface samples: no mesh, no polyline, no discrete points"
                    : rec.invalidReason;
                return rec;
            }

            // Pose uncertainty large relative to the feature scale makes the ray casts meaningless.
            if (rec.targetExtent > 0f && vp.poseUncertaintyM > 0.5f * Mathf.Sqrt(Mathf.Max(rec.targetExtent, 1e-6f)))
            {
                rec.conditionValid = false;
                rec.invalidReason = $"pose uncertainty {vp.poseUncertaintyM:F2} m is large relative to " +
                                    $"the target extent {rec.targetExtent:F2} {rec.extentUnit}";
            }

            BuildFrustumPlanes(vp, out Plane[] planes);

            float weightTotal = 0f;
            float weightInFrustum = 0f;
            float weightVisible = 0f;

            float distanceAccum = 0f;
            float incidenceAccum = 0f;
            float nearest = float.MaxValue;
            int visibleCount = 0;

            var visible = new bool[samples.Count];

            for (int si = 0; si < samples.Count; si++)
            {
                var s = samples[si];
                weightTotal += s.Weight;

                if (!InsideFrustum(planes, s.Position)) continue;
                weightInFrustum += s.Weight;

                Vector3 toCamera = vp.position - s.Position;
                float distance = toCamera.magnitude;
                if (distance < 1e-4f) continue;
                Vector3 dir = toCamera / distance;

                // Back-face test: a sample whose outward normal points away from the camera
                // is on the far side of the target and is not observable from here.
                float facing = Vector3.Dot(s.Normal, dir);
                if (backFaceCounts && facing <= 0f) continue;

                // Line-of-sight test against the occluder set.
                Vector3 origin = s.Position + s.Normal * rayOriginEpsilon;
                if (Physics.Raycast(origin, dir, distance - rayOriginEpsilon * 2f, occluderMask,
                                    QueryTriggerInteraction.Ignore))
                    continue;

                visible[si] = true;
                weightVisible += s.Weight;
                visibleCount++;
                distanceAccum += distance;
                incidenceAccum += Mathf.Acos(Mathf.Clamp(Mathf.Abs(facing), 0f, 1f)) * Mathf.Rad2Deg;
                if (distance < nearest) nearest = distance;
            }

            rec.visibleSampleMask = EncodeMask(visible);
            rec.visiblePointCount = visibleCount;
            rec.frustumCoverage = weightTotal > 0f ? weightInFrustum / weightTotal : 0f;
            rec.unoccludedFraction = weightInFrustum > 0f ? weightVisible / weightInFrustum : 0f;
            rec.effectiveCoverage = weightTotal > 0f ? weightVisible / weightTotal : 0f;

            if (visibleCount > 0)
            {
                rec.viewingDistanceM = distanceAccum / visibleCount;
                rec.incidenceAngleDeg = incidenceAccum / visibleCount;
                rec.nearestDistanceM = nearest;

                // Ground sample distance perpendicular to the line of sight.
                float halfFov = vp.horizontalFovDeg * 0.5f * Mathf.Deg2Rad;
                rec.gsdPerpendicularM = 2f * rec.viewingDistanceM * Mathf.Tan(halfFov) / Mathf.Max(vp.imageWidthPx, 1);

                // Foreshortening: at grazing incidence the same pixel spans far more surface,
                // which is why a barrier photographed end-on can be at ten metres and still
                // be unresolvable.
                float cosInc = Mathf.Cos(rec.incidenceAngleDeg * Mathf.Deg2Rad);
                rec.gsdAlongSurfaceM = rec.gsdPerpendicularM / Mathf.Max(cosInc, 0.05f);
            }
            else
            {
                rec.viewingDistanceM = 0f;
                rec.nearestDistanceM = 0f;
                rec.incidenceAngleDeg = 90f;
                rec.gsdPerpendicularM = 0f;
                rec.gsdAlongSurfaceM = 0f;
            }

            return rec;
        }

        // -- Mask encoding --------------------------------------------------

        /// <summary>
        /// Encode per-sample visibility as a little-endian hex string: sample i occupies bit i,
        /// so nibble k of the string holds samples 4k to 4k+3. The Python adapter decodes with
        /// int(mask, 16) and shifts, so the two ends must agree on bit order — they do here, and
        /// the round trip is covered by the test suite.
        /// </summary>
        private static string EncodeMask(bool[] visible)
        {
            if (visible == null || visible.Length == 0) return "";

            int nibbles = (visible.Length + 3) / 4;
            var chars = new char[nibbles];
            for (int n = 0; n < nibbles; n++)
            {
                int value = 0;
                for (int b = 0; b < 4; b++)
                {
                    int idx = n * 4 + b;
                    if (idx < visible.Length && visible[idx]) value |= 1 << b;
                }
                chars[nibbles - 1 - n] = "0123456789abcdef"[value];
            }
            return new string(chars);
        }

        // -- Frustum --------------------------------------------------------

        /// <summary>
        /// Builds the six frustum planes for a viewpoint without instantiating a Camera, so the
        /// analysis can run headless in batch mode.
        /// </summary>
        private static void BuildFrustumPlanes(Viewpoint vp, out Plane[] planes)
        {
            float aspect = (float)vp.imageWidthPx / Mathf.Max(vp.imageHeightPx, 1);
            float halfH = vp.horizontalFovDeg * 0.5f * Mathf.Deg2Rad;
            // Unity's Camera.fieldOfView is vertical; convert from the horizontal FOV we store.
            float vFovRad = 2f * Mathf.Atan(Mathf.Tan(halfH) / aspect);

            Matrix4x4 proj = Matrix4x4.Perspective(vFovRad * Mathf.Rad2Deg, aspect, 0.05f, 5000f);
            Matrix4x4 view = Matrix4x4.TRS(vp.position, vp.rotation, new Vector3(1, 1, -1)).inverse;
            planes = GeometryUtility.CalculateFrustumPlanes(proj * view);
        }

        private static bool InsideFrustum(Plane[] planes, Vector3 point)
        {
            for (int i = 0; i < planes.Length; i++)
                if (planes[i].GetDistanceToPoint(point) < 0f) return false;
            return true;
        }

        // -- Batch ----------------------------------------------------------

        /// <summary>
        /// Analyse every marked target against every viewpoint. Records with zero effective
        /// coverage are retained rather than dropped: "this viewpoint saw none of this target"
        /// is exactly the information the sufficiency model needs, and discarding it would
        /// reintroduce the silent gap the framework exists to close.
        /// </summary>
        public List<ObservationConditionRecord> AnalyseAll(
            IEnumerable<InspectionTargetMarker> targets,
            IEnumerable<Viewpoint> viewpoints)
        {
            var results = new List<ObservationConditionRecord>();
            foreach (var t in targets)
            {
                if (t == null || string.IsNullOrEmpty(t.targetLocalName)) continue;
                foreach (var vp in viewpoints)
                    results.Add(Analyse(t, vp));
            }
            return results;
        }
    }
}

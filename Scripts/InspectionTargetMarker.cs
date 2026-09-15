using System.Collections.Generic;
using UnityEngine;

namespace Cieo.Visibility
{
    /// <summary>
    /// How a target's extent is measured, and therefore how coverage over it is defined.
    /// </summary>
    public enum TargetExtentKind
    {
        /// <summary>Extent is a length in metres along a polyline: an excavation perimeter,
        /// a guardrail alignment, a work-zone boundary. Coverage is the fraction of that
        /// length observed.</summary>
        Linear,

        /// <summary>Extent is an area in square metres over a mesh surface: a deck slab,
        /// a pavement, an excavation face. Coverage is the fraction of that area observed.</summary>
        Surface,

        /// <summary>Extent is a count of discrete sub-features: the bolts in a connection,
        /// the studs in an array. Coverage is the fraction of those features observed.</summary>
        Discrete
    }

    /// <summary>
    /// Marks a GameObject in the reconstructed scene as something an inspection requirement
    /// can be predicated on, and tells the analyser how to sample it.
    ///
    /// The marker carries the CIEO target IRI so that the analysis output joins straight back
    /// into the knowledge graph without a lookup table. Attach one to every excavation
    /// perimeter, exclusion zone, guardrail run and connection assembly that appears in a
    /// phase scene.
    /// </summary>
    [DisallowMultipleComponent]
    public class InspectionTargetMarker : MonoBehaviour
    {
        [Header("Knowledge graph identity")]
        [Tooltip("Local name of the target IRI in the CIEO graph, e.g. 'ExcavationBoundary_North'. " +
                 "The batch runner prefixes this with the case-study namespace.")]
        public string targetLocalName = "";

        [Tooltip("CIEO class of this target, e.g. 'cst:ExcavationBoundary'. Written into the output " +
                 "so the adapter can sanity-check the graph against the scene.")]
        public string cieoClass = "";

        [Header("Extent")]
        public TargetExtentKind extentKind = TargetExtentKind.Surface;

        [Tooltip("For Linear targets: the polyline, in local space, whose length is the extent. " +
                 "Leave empty to derive it from a LineRenderer on the same GameObject.")]
        public List<Vector3> polylineLocal = new List<Vector3>();

        [Tooltip("For Discrete targets: the sub-feature positions, in local space — one per bolt, " +
                 "stud or fixing whose presence the requirement depends on.")]
        public List<Vector3> discretePointsLocal = new List<Vector3>();

        [Tooltip("Surface normal to assume for Discrete sub-features, in local space. Fasteners are " +
                 "usually observed along the axis of the hole, which is what determines whether the " +
                 "washer edge is visible at all.")]
        public Vector3 discreteNormalLocal = Vector3.up;

        [Header("Sampling")]
        [Tooltip("Number of surface samples used to estimate coverage. Coverage figures are Monte " +
                 "Carlo estimates: the standard error is roughly 0.5/sqrt(n), so 256 samples give " +
                 "about +/-3% and 1024 about +/-1.5%. Report whichever you use.")]
        [Range(16, 8192)]
        public int sampleCount = 512;

        [Tooltip("Fixed seed, so a reported coverage figure can be reproduced exactly.")]
        public int samplingSeed = 20260910;

        [Header("Diagnostics")]
        public bool drawGizmos = true;

        private Mesh _cachedMesh;
        private float[] _cumulativeTriangleArea;
        private Vector3[] _cachedVertices;
        private int[] _cachedTriangles;

        /// <summary>
        /// The magnitude of the target's extent, in metres, square metres, or count,
        /// according to <see cref="extentKind"/>. This is the denominator of every
        /// coverage figure the analyser reports.
        /// </summary>
        public float ComputeExtent()
        {
            switch (extentKind)
            {
                case TargetExtentKind.Linear:
                {
                    var pts = ResolvePolylineWorld();
                    float total = 0f;
                    for (int i = 1; i < pts.Count; i++) total += Vector3.Distance(pts[i - 1], pts[i]);
                    return total;
                }
                case TargetExtentKind.Discrete:
                    return discretePointsLocal.Count;
                default:
                {
                    EnsureMeshCache();
                    if (_cumulativeTriangleArea == null || _cumulativeTriangleArea.Length == 0) return 0f;
                    return _cumulativeTriangleArea[_cumulativeTriangleArea.Length - 1];
                }
            }
        }

        public string ExtentUnit()
        {
            switch (extentKind)
            {
                case TargetExtentKind.Linear:   return "m";
                case TargetExtentKind.Discrete: return "count";
                default:                        return "m2";
            }
        }

        /// <summary>
        /// Draws the sample set used for a visibility estimate: world positions with outward
        /// normals. Deterministic for a given seed and count.
        /// </summary>
        public List<SurfaceSample> Sample()
        {
            var rng = new System.Random(samplingSeed);
            var samples = new List<SurfaceSample>(sampleCount);

            switch (extentKind)
            {
                case TargetExtentKind.Linear:
                    SampleLinear(samples);
                    break;
                case TargetExtentKind.Discrete:
                    SampleDiscrete(samples);
                    break;
                default:
                    SampleSurface(samples, rng);
                    break;
            }
            return samples;
        }

        // -- Linear ---------------------------------------------------------

        private List<Vector3> ResolvePolylineWorld()
        {
            var world = new List<Vector3>();
            if (polylineLocal != null && polylineLocal.Count >= 2)
            {
                foreach (var p in polylineLocal) world.Add(transform.TransformPoint(p));
                return world;
            }

            var lr = GetComponent<LineRenderer>();
            if (lr != null && lr.positionCount >= 2)
            {
                var buf = new Vector3[lr.positionCount];
                lr.GetPositions(buf);
                foreach (var p in buf)
                    world.Add(lr.useWorldSpace ? p : transform.TransformPoint(p));
                return world;
            }

            // Fall back to the footprint of the renderer bounds. Crude, and the analyser flags
            // it, because a coverage figure over a guessed extent is not a coverage figure.
            var rend = GetComponent<Renderer>();
            if (rend != null)
            {
                var b = rend.bounds;
                world.Add(new Vector3(b.min.x, b.center.y, b.min.z));
                world.Add(new Vector3(b.max.x, b.center.y, b.min.z));
                world.Add(new Vector3(b.max.x, b.center.y, b.max.z));
                world.Add(new Vector3(b.min.x, b.center.y, b.max.z));
                world.Add(new Vector3(b.min.x, b.center.y, b.min.z));
            }
            return world;
        }

        /// <summary>True when the polyline had to be guessed from bounds rather than authored.</summary>
        public bool PolylineIsDerived =>
            extentKind == TargetExtentKind.Linear
            && (polylineLocal == null || polylineLocal.Count < 2)
            && GetComponent<LineRenderer>() == null;

        private void SampleLinear(List<SurfaceSample> samples)
        {
            var pts = ResolvePolylineWorld();
            if (pts.Count < 2) return;

            var segLen = new float[pts.Count - 1];
            float total = 0f;
            for (int i = 1; i < pts.Count; i++)
            {
                segLen[i - 1] = Vector3.Distance(pts[i - 1], pts[i]);
                total += segLen[i - 1];
            }
            if (total <= 1e-6f) return;

            // Uniform by arc length: each sample represents an equal share of the extent,
            // so the fraction of samples visible is directly the fraction of length observed.
            for (int s = 0; s < sampleCount; s++)
            {
                float t = (s + 0.5f) / sampleCount * total;
                float acc = 0f;
                for (int i = 0; i < segLen.Length; i++)
                {
                    if (acc + segLen[i] >= t || i == segLen.Length - 1)
                    {
                        float local = segLen[i] > 1e-6f ? (t - acc) / segLen[i] : 0f;
                        Vector3 pos = Vector3.Lerp(pts[i], pts[i + 1], Mathf.Clamp01(local));
                        Vector3 tangent = (pts[i + 1] - pts[i]).normalized;
                        // For a protective barrier the meaningful normal is horizontal and
                        // perpendicular to the run: that is the face an inspector photographs.
                        Vector3 normal = Vector3.Cross(Vector3.up, tangent).normalized;
                        if (normal.sqrMagnitude < 1e-6f) normal = Vector3.up;
                        samples.Add(new SurfaceSample(pos + Vector3.up * 0.5f, normal, total / sampleCount));
                        break;
                    }
                    acc += segLen[i];
                }
            }
        }

        // -- Discrete -------------------------------------------------------

        private void SampleDiscrete(List<SurfaceSample> samples)
        {
            Vector3 n = transform.TransformDirection(discreteNormalLocal).normalized;
            foreach (var p in discretePointsLocal)
                samples.Add(new SurfaceSample(transform.TransformPoint(p), n, 1f));
        }

        // -- Surface --------------------------------------------------------

        private void EnsureMeshCache()
        {
            var mf = GetComponent<MeshFilter>();
            Mesh mesh = mf != null ? mf.sharedMesh : null;
            if (mesh == null)
            {
                var smr = GetComponent<SkinnedMeshRenderer>();
                if (smr != null) mesh = smr.sharedMesh;
            }
            if (mesh == null || mesh == _cachedMesh) return;

            _cachedMesh = mesh;
            _cachedVertices = mesh.vertices;
            _cachedTriangles = mesh.triangles;

            int triCount = _cachedTriangles.Length / 3;
            _cumulativeTriangleArea = new float[triCount];
            float acc = 0f;
            for (int t = 0; t < triCount; t++)
            {
                Vector3 a = transform.TransformPoint(_cachedVertices[_cachedTriangles[t * 3]]);
                Vector3 b = transform.TransformPoint(_cachedVertices[_cachedTriangles[t * 3 + 1]]);
                Vector3 c = transform.TransformPoint(_cachedVertices[_cachedTriangles[t * 3 + 2]]);
                acc += Vector3.Cross(b - a, c - a).magnitude * 0.5f;
                _cumulativeTriangleArea[t] = acc;
            }
        }

        private void SampleSurface(List<SurfaceSample> samples, System.Random rng)
        {
            EnsureMeshCache();
            if (_cumulativeTriangleArea == null || _cumulativeTriangleArea.Length == 0) return;

            float totalArea = _cumulativeTriangleArea[_cumulativeTriangleArea.Length - 1];
            if (totalArea <= 1e-9f) return;
            float weight = totalArea / sampleCount;

            for (int s = 0; s < sampleCount; s++)
            {
                // Area-weighted triangle choice, so samples are uniform over the surface
                // rather than uniform over the triangle list.
                float r = (float)rng.NextDouble() * totalArea;
                int lo = 0, hi = _cumulativeTriangleArea.Length - 1;
                while (lo < hi)
                {
                    int mid = (lo + hi) / 2;
                    if (_cumulativeTriangleArea[mid] < r) lo = mid + 1; else hi = mid;
                }

                Vector3 a = transform.TransformPoint(_cachedVertices[_cachedTriangles[lo * 3]]);
                Vector3 b = transform.TransformPoint(_cachedVertices[_cachedTriangles[lo * 3 + 1]]);
                Vector3 c = transform.TransformPoint(_cachedVertices[_cachedTriangles[lo * 3 + 2]]);

                float u = (float)rng.NextDouble();
                float v = (float)rng.NextDouble();
                if (u + v > 1f) { u = 1f - u; v = 1f - v; }
                Vector3 pos = a + u * (b - a) + v * (c - a);
                Vector3 normal = Vector3.Cross(b - a, c - a).normalized;

                samples.Add(new SurfaceSample(pos, normal, weight));
            }
        }

        private void OnDrawGizmosSelected()
        {
            if (!drawGizmos) return;
            var samples = Sample();
            Gizmos.color = new Color(0.2f, 0.8f, 1f, 0.7f);
            foreach (var s in samples)
            {
                Gizmos.DrawSphere(s.Position, 0.05f);
                Gizmos.DrawRay(s.Position, s.Normal * 0.25f);
            }
        }
    }

    /// <summary>A point on the target, its outward normal, and the share of the extent it stands for.</summary>
    public struct SurfaceSample
    {
        public readonly Vector3 Position;
        public readonly Vector3 Normal;
        public readonly float Weight;

        public SurfaceSample(Vector3 position, Vector3 normal, float weight)
        {
            Position = position;
            Normal = normal;
            Weight = weight;
        }
    }
}

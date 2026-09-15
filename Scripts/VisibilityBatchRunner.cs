using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Cieo.Visibility
{
    /// <summary>Wrapper types, because Unity's JsonUtility will not serialise a bare array.</summary>
    [Serializable] public class ViewpointFile { public List<ViewpointDto> viewpoints = new List<ViewpointDto>(); }

    [Serializable]
    public class ViewpointDto
    {
        public string viewpoint_id;
        public string image_id;
        public string captured_at;

        // Camera centre in the reconstruction's world frame.
        public float tx, ty, tz;

        // Rotation as a quaternion in the reconstruction's world frame.
        public float qx, qy, qz, qw;

        public float horizontal_fov_deg = 60f;
        public int image_width_px = 4000;
        public int image_height_px = 3000;
        public string pose_source = "estimated";
        public float pose_uncertainty_m = 0.25f;
    }

    [Serializable]
    public class ObservationConditionFile
    {
        public string analysis_engine;
        public string analysis_version;
        public string geometry_source;
        public string occluder_set;
        public string generated_at;
        public string scene_name;
        public string phase_id;
        public int ray_count;
        public List<ObservationConditionRecord> conditions = new List<ObservationConditionRecord>();
    }

    /// <summary>
    /// Drives the visibility analysis over a phase scene and writes the observation conditions
    /// out as JSON for <c>src/visibility_to_rdf.py</c> to lift into the knowledge graph.
    ///
    /// Put one of these in each phase scene, point it at the pose file the reconstruction
    /// pipeline produced, and run it. Nothing here needs the editor: <see cref="Run"/> works in
    /// batch mode, which is what makes the coverage figures in a paper reproducible from a
    /// command line rather than from a sequence of clicks.
    /// </summary>
    public class VisibilityBatchRunner : MonoBehaviour
    {
        [Header("Scene identity")]
        [Tooltip("CIEO phase this scene represents, e.g. 'P3'. Written into the output so the " +
                 "adapter can scope the conditions to the right phase.")]
        public string phaseId = "P3";

        [Header("Input")]
        [Tooltip("JSON file of camera poses from the depth-estimation and registration pipeline. " +
                 "Relative paths resolve against the project folder.")]
        public string viewpointFilePath = "Data/viewpoints_P3.json";

        [Header("Output")]
        public string outputFilePath = "Data/observation_conditions_P3.json";

        [Header("Coordinate frame")]
        [Tooltip("Reconstruction pipelines generally emit a right-handed, Z-up or Y-down frame; " +
                 "Unity is left-handed and Y-up. Getting this wrong does not throw — it silently " +
                 "produces plausible coverage numbers for the wrong geometry, so verify by " +
                 "checking that a known camera lands where it was standing before trusting a run.")]
        public CoordinateConvention sourceConvention = CoordinateConvention.RightHandedYUp;

        [Tooltip("Similarity transform from the reconstruction frame to the Unity scene frame, " +
                 "applied after the handedness conversion. Scale should already be metric if the " +
                 "point cloud was scaled against on-site reference measurements.")]
        public Vector3 frameTranslation = Vector3.zero;
        public Vector3 frameRotationEuler = Vector3.zero;
        public float frameScale = 1f;

        public enum CoordinateConvention
        {
            /// <summary>Already in Unity's frame; no conversion.</summary>
            UnityLeftHandedYUp,
            /// <summary>Right-handed, Y up (OpenGL-like): negate Z.</summary>
            RightHandedYUp,
            /// <summary>Right-handed, Z up (COLMAP world / survey-like): swap Y and Z.</summary>
            RightHandedZUp
        }

        public VisibilityAnalyzer analyzer;

        [ContextMenu("Run visibility analysis")]
        public void Run()
        {
            if (analyzer == null) analyzer = GetComponent<VisibilityAnalyzer>();
            if (analyzer == null)
            {
                Debug.LogError("[CIEO] No VisibilityAnalyzer assigned or attached.");
                return;
            }

            var viewpoints = LoadViewpoints(Resolve(viewpointFilePath));
            if (viewpoints.Count == 0)
            {
                Debug.LogError($"[CIEO] No viewpoints loaded from {viewpointFilePath}.");
                return;
            }

            var targets = new List<InspectionTargetMarker>(
                FindObjectsByType<InspectionTargetMarker>(FindObjectsSortMode.None));
            if (targets.Count == 0)
            {
                Debug.LogError("[CIEO] No InspectionTargetMarker components found in the scene.");
                return;
            }

            var records = analyzer.AnalyseAll(targets, viewpoints);

            var file = new ObservationConditionFile
            {
                analysis_engine = analyzer.analysisEngine,
                analysis_version = analyzer.analysisVersion,
                geometry_source = analyzer.geometrySource,
                occluder_set = analyzer.occluderSetDescription,
                generated_at = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"),
                scene_name = gameObject.scene.name,
                phase_id = phaseId,
                ray_count = records.Count,
                conditions = records
            };

            string outPath = Resolve(outputFilePath);
            Directory.CreateDirectory(Path.GetDirectoryName(outPath));
            File.WriteAllText(outPath, JsonUtility.ToJson(file, true));

            int invalid = 0, zeroCoverage = 0;
            foreach (var r in records)
            {
                if (!r.conditionValid) invalid++;
                if (r.effectiveCoverage <= 0f) zeroCoverage++;
            }

            Debug.Log($"[CIEO] {records.Count} observation conditions " +
                      $"({targets.Count} targets x {viewpoints.Count} viewpoints) written to {outPath}. " +
                      $"{invalid} flagged invalid, {zeroCoverage} with zero effective coverage.");
        }

        private string Resolve(string path) =>
            Path.IsPathRooted(path) ? path : Path.Combine(Application.dataPath, "..", path);

        private List<Viewpoint> LoadViewpoints(string path)
        {
            var list = new List<Viewpoint>();
            if (!File.Exists(path))
            {
                Debug.LogError($"[CIEO] Viewpoint file not found: {path}");
                return list;
            }

            var dto = JsonUtility.FromJson<ViewpointFile>(File.ReadAllText(path));
            if (dto == null || dto.viewpoints == null) return list;

            Quaternion frameRot = Quaternion.Euler(frameRotationEuler);

            foreach (var v in dto.viewpoints)
            {
                ConvertPose(v, out Vector3 pos, out Quaternion rot);
                pos = frameRot * (pos * frameScale) + frameTranslation;
                rot = frameRot * rot;

                list.Add(new Viewpoint
                {
                    viewpointId = v.viewpoint_id,
                    imageId = v.image_id,
                    capturedAt = v.captured_at,
                    position = pos,
                    rotation = rot,
                    horizontalFovDeg = v.horizontal_fov_deg,
                    imageWidthPx = v.image_width_px,
                    imageHeightPx = v.image_height_px,
                    poseSource = v.pose_source,
                    poseUncertaintyM = v.pose_uncertainty_m
                });
            }
            return list;
        }

        private void ConvertPose(ViewpointDto v, out Vector3 pos, out Quaternion rot)
        {
            switch (sourceConvention)
            {
                case CoordinateConvention.RightHandedYUp:
                    // Mirror the Z axis: position negates, and the quaternion's x and y
                    // components negate with it.
                    pos = new Vector3(v.tx, v.ty, -v.tz);
                    rot = new Quaternion(-v.qx, -v.qy, v.qz, v.qw);
                    break;

                case CoordinateConvention.RightHandedZUp:
                    // Swap Y and Z, which also mirrors handedness.
                    pos = new Vector3(v.tx, v.tz, v.ty);
                    rot = new Quaternion(-v.qx, -v.qz, -v.qy, v.qw);
                    break;

                default:
                    pos = new Vector3(v.tx, v.ty, v.tz);
                    rot = new Quaternion(v.qx, v.qy, v.qz, v.qw);
                    break;
            }
            rot = Quaternion.Normalize(rot);
        }
    }
}

using System.Collections.Generic;
using UnityEngine;

namespace Cieo.Visibility
{
    /// <summary>
    /// Marks WHERE ON SITE a photograph was taken from.
    ///
    /// This is the one piece of information the current setup is missing, and everything else
    /// depends on it. Assigning a photo to a 3D model says "this picture is of that thing". It
    /// does not say where the photographer stood, and without that the system cannot work out
    /// what the picture could and could not have shown — which is the whole question.
    ///
    /// HOW TO PLACE ONE
    /// ---------------
    ///   1. Create an empty GameObject. Name it after the photo, e.g. "VP_IMG_047".
    ///   2. Add this component and drag the photo into the Photo field.
    ///   3. Tick "Show photo overlay" and move the GameObject in the Scene view until the
    ///      overlaid photo lines up with your 3D model. Rotate it, walk it forward and back.
    ///      When the bridge edges and the ground line in the photo sit on top of the same edges
    ///      in the model, the pose is right.
    ///   4. Untick the overlay.
    ///
    /// That is it. Ten to fifteen minutes of practice and each photo takes about a minute. You do
    /// NOT need survey equipment and you do NOT need the depth-estimation poses for this — you
    /// modelled the site, so you can put the camera back where you stood. Record how confident
    /// you are in Pose Uncertainty and the reasoning will carry that uncertainty through.
    /// </summary>
    [DisallowMultipleComponent]
    public class PhotoViewpoint : MonoBehaviour
    {
        [Header("Identity")]
        [Tooltip("Must match the image_id used in the detector output, e.g. 'IMG_047'.")]
        public string imageId = "";

        [Tooltip("Leave blank to use this GameObject's name.")]
        public string viewpointId = "";

        [Tooltip("ISO date, e.g. 2026-04-14T09:31:00Z. Site conditions change; evidence has an age.")]
        public string capturedAt = "";

        [Header("The photograph")]
        public Texture2D photo;

        [Tooltip("Path relative to the project, for the export. Optional.")]
        public string photoPath = "";

        [Header("Camera")]
        [Tooltip("Horizontal field of view of the camera that took the photo. A phone's main lens " +
                 "is usually 65-70 degrees; a DSLR at 24mm on full frame is about 74; at 50mm " +
                 "about 40. Getting this wrong scales every resolution estimate, so check it " +
                 "against the EXIF rather than guessing.")]
        public float horizontalFovDeg = 67f;

        [Tooltip("Pixel dimensions of the original photo, not of any resized copy.")]
        public int imageWidthPx = 4032;
        public int imageHeightPx = 3024;

        [Header("How sure are you of this pose?")]
        [Tooltip("Roughly how far out the position might be, in metres. Be honest: 0.5 m for a " +
                 "careful overlay against clear building edges, 2 m for an open area with nothing " +
                 "to align against. This propagates into every decision the photo supports, and " +
                 "the analyser refuses to trust a pose that is loose relative to the target size.")]
        public float poseUncertaintyM = 0.5f;

        [Header("Alignment helper")]
        [Tooltip("Draws the photo over the Game view at half opacity so you can line the model up " +
                 "with it. Turn it off when you are done.")]
        public bool showPhotoOverlay = false;

        [Range(0f, 1f)] public float overlayOpacity = 0.5f;

        [Header("What this photo is of (optional)")]
        [Tooltip("Purely documentation: the analyser tests this viewpoint against every target " +
                 "in the scene anyway, and a photo showing a target you did not expect is useful " +
                 "evidence rather than a mistake.")]
        public List<InspectionTargetMarker> intendedSubjects = new List<InspectionTargetMarker>();

        private Camera _previewCamera;

        public string ResolvedViewpointId =>
            string.IsNullOrEmpty(viewpointId) ? gameObject.name : viewpointId;

        public string ResolvedImageId =>
            string.IsNullOrEmpty(imageId) ? (photo != null ? photo.name : gameObject.name) : imageId;

        /// <summary>Convert to the pose record the analyser consumes.</summary>
        public Viewpoint ToViewpoint()
        {
            return new Viewpoint
            {
                viewpointId = ResolvedViewpointId,
                imageId = ResolvedImageId,
                capturedAt = capturedAt,
                position = transform.position,
                rotation = transform.rotation,
                horizontalFovDeg = horizontalFovDeg,
                imageWidthPx = imageWidthPx,
                imageHeightPx = imageHeightPx,
                // Poses placed by hand in the model are neither surveyed nor reconstructed. They
                // are estimated, and labelling them honestly is what keeps the pose uncertainty
                // from being quietly forgotten downstream.
                poseSource = "estimated",
                poseUncertaintyM = poseUncertaintyM,
            };
        }

        /// <summary>
        /// Drop a camera at this pose so the Game view shows what the photographer saw.
        /// Use it with the overlay to check the alignment.
        /// </summary>
        [ContextMenu("Preview through this viewpoint")]
        public void Preview()
        {
            if (_previewCamera == null)
            {
                var go = new GameObject($"[preview] {ResolvedViewpointId}");
                go.transform.SetParent(transform, false);
                _previewCamera = go.AddComponent<Camera>();
                _previewCamera.depth = 100;
            }

            float aspect = (float)imageWidthPx / Mathf.Max(imageHeightPx, 1);
            _previewCamera.aspect = aspect;
            // Unity's fieldOfView is vertical; the photo's is horizontal.
            _previewCamera.fieldOfView =
                2f * Mathf.Atan(Mathf.Tan(horizontalFovDeg * 0.5f * Mathf.Deg2Rad) / aspect) * Mathf.Rad2Deg;
            _previewCamera.enabled = true;
            showPhotoOverlay = true;
        }

        [ContextMenu("Stop preview")]
        public void StopPreview()
        {
            showPhotoOverlay = false;
            if (_previewCamera != null) _previewCamera.enabled = false;
        }

        private void OnGUI()
        {
            if (!showPhotoOverlay || photo == null) return;
            var prev = GUI.color;
            GUI.color = new Color(1f, 1f, 1f, overlayOpacity);
            GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), photo, ScaleMode.ScaleToFit);
            GUI.color = prev;

            GUI.Label(new Rect(10, 10, 600, 60),
                $"Aligning {ResolvedViewpointId} — move and rotate this GameObject until the " +
                $"photo lines up with the model, then untick Show Photo Overlay.");
        }

        private void OnDrawGizmos()
        {
            Gizmos.color = new Color(1f, 0.85f, 0.2f, 0.9f);
            Gizmos.DrawSphere(transform.position, 0.25f);

            // Draw the frustum so you can see at a glance what each photo covers, and spot
            // targets that no photo points at.
            float aspect = (float)imageWidthPx / Mathf.Max(imageHeightPx, 1);
            float vFov = 2f * Mathf.Atan(Mathf.Tan(horizontalFovDeg * 0.5f * Mathf.Deg2Rad) / aspect)
                         * Mathf.Rad2Deg;
            Matrix4x4 old = Gizmos.matrix;
            Gizmos.matrix = Matrix4x4.TRS(transform.position, transform.rotation, Vector3.one);
            Gizmos.DrawFrustum(Vector3.zero, vFov, 18f, 0.3f, aspect);
            Gizmos.matrix = old;
        }
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

namespace Cieo.Visibility
{
    [Serializable]
    public class DecisionRecord
    {
        public string item_id;
        public string target_id;
        public string requirement_id;
        public string phase;
        public string requirement_statement;
        public string evidence_pattern;
        public string state;
        public string rule_id;
        public string regulation;
        public string clause;
        public string rationale;
        public string recommended_action;
        public bool requires_human_verification;
        public bool trace_complete;
        public float aggregated_recall;
        public float posterior_absence;
        public float coverage_achieved;
        public float coverage_required;
        public string sufficiency_level;
        public string limiting_class;
        public int n_viewpoints;
        public List<string> viewpoint_ids = new List<string>();
        public List<string> image_ids = new List<string>();
        public List<string> supporting_detections = new List<string>();
    }

    [Serializable]
    public class DecisionFile
    {
        public string generated_at;
        public string sufficiency_model;
        public List<DecisionRecord> decisions = new List<DecisionRecord>();
    }

    /// <summary>
    /// Replaces the "here is the annotated photo, you decide" panel.
    ///
    /// What changes for the inspector
    /// ------------------------------
    /// Before: the panel showed a photo with boxes drawn on it and left the inspector to work out
    /// what it meant for compliance. Every single item needed a human judgement, and the system
    /// contributed nothing beyond drawing the boxes.
    ///
    /// After: the panel states which regulation applies, what the system concluded, and — where it
    /// concluded nothing — why the photograph could not settle the question and what would. The
    /// inspector's attention goes to the items that actually need it instead of to all of them.
    ///
    /// The distinction the panel exists to make visible is between the last two of these:
    ///
    ///   COMPLIANT               the requirement is met, on evidence good enough to say so
    ///   NON-COMPLIANT           the requirement is breached, on evidence good enough to say so
    ///   INSUFFICIENT EVIDENCE   nothing was detected, and nothing would have been even if it
    ///                           were there. This is a finding about the photograph.
    ///   NEEDS HUMAN CHECK       the evidence is marginal; the inspector decides
    ///   OUT OF SCOPE            no photograph will ever settle this; it needs a certificate,
    ///                           a test result or a survey
    ///
    /// Wiring: put this on your existing inspection panel, assign the Text fields, and call
    /// ShowFor(targetId) from the click handler that currently opens the photo.
    /// </summary>
    public class InspectionResultPanel : MonoBehaviour
    {
        [Header("Data")]
        [Tooltip("File name inside StreamingAssets. Produced by src/run_reasoning.py as " +
                 "decisions.json — copy it into Assets/StreamingAssets/.")]
        public string decisionsFileName = "decisions.json";

        [Header("UI — assign your existing panel's fields")]
        public GameObject panelRoot;
        public Text titleText;
        public Text stateText;
        public Text regulationText;
        public Text rationaleText;
        public Text actionText;
        public Text evidenceText;
        public Image stateSwatch;
        public RawImage photoView;

        [Header("Photos")]
        [Tooltip("Site photographs, keyed by the image_id used in the detector output. The panel " +
                 "shows the photographs the decision was actually based on, not an arbitrary one.")]
        public List<Texture2D> photos = new List<Texture2D>();

        [Header("State colours")]
        public Color compliantColor = new Color(0.16f, 0.62f, 0.33f);
        public Color nonCompliantColor = new Color(0.78f, 0.21f, 0.18f);
        public Color insufficientColor = new Color(0.55f, 0.52f, 0.48f);
        public Color humanCheckColor = new Color(0.85f, 0.60f, 0.13f);
        public Color outOfScopeColor = new Color(0.35f, 0.40f, 0.52f);

        private DecisionFile _file;
        private readonly Dictionary<string, List<DecisionRecord>> _byTarget = new();
        private readonly Dictionary<string, Texture2D> _photoByName = new();

        private void Awake()
        {
            LoadDecisions();
            foreach (var p in photos)
                if (p != null) _photoByName[p.name] = p;
            if (panelRoot != null) panelRoot.SetActive(false);
        }

        private void LoadDecisions()
        {
            string path = Path.Combine(Application.streamingAssetsPath, decisionsFileName);
            if (!File.Exists(path))
            {
                Debug.LogError($"[CIEO] {path} not found. Run src/run_reasoning.py and copy " +
                               $"evaluation/results/decisions.json into Assets/StreamingAssets/.");
                return;
            }

            _file = JsonUtility.FromJson<DecisionFile>(File.ReadAllText(path));
            _byTarget.Clear();
            foreach (var d in _file.decisions)
            {
                if (!_byTarget.TryGetValue(d.target_id, out var list))
                    _byTarget[d.target_id] = list = new List<DecisionRecord>();
                list.Add(d);
            }
            Debug.Log($"[CIEO] Loaded {_file.decisions.Count} decisions across {_byTarget.Count} " +
                      $"targets (model: {_file.sufficiency_model}).");
        }

        /// <summary>
        /// Call this from the click handler that currently opens the photo. Pass the target's
        /// local name — the same string you put in that model's InspectionTargetMarker.
        /// </summary>
        public void ShowFor(string targetLocalName)
        {
            if (panelRoot != null) panelRoot.SetActive(true);

            if (_byTarget == null || !_byTarget.TryGetValue(targetLocalName, out var records)
                || records.Count == 0)
            {
                SetText(titleText, targetLocalName);
                SetText(stateText, "NO REQUIREMENT APPLIES");
                SetText(regulationText, "");
                SetText(rationaleText,
                    "No inspection requirement is scoped to this object in the active construction " +
                    "phase. That is not a gap in the system: obligations bind during the phases " +
                    "they belong to, and this one does not bind now.");
                SetText(actionText, "");
                SetText(evidenceText, "");
                if (stateSwatch != null) stateSwatch.color = outOfScopeColor;
                return;
            }

            // Show the most serious outstanding item first — that is where attention belongs.
            records.Sort((a, b) => Severity(a.state).CompareTo(Severity(b.state)));
            Render(records[0], records.Count);
        }

        private static int Severity(string state) => state switch
        {
            "NonCompliant" => 0,
            "InsufficientEvidence" => 1,
            "RequiresHumanVerification" => 2,
            "OutOfAutomaticScope" => 3,
            "Compliant" => 4,
            _ => 5,
        };

        private void Render(DecisionRecord d, int totalForTarget)
        {
            SetText(titleText, $"{d.target_id}   ·   {d.requirement_id}"
                               + (totalForTarget > 1 ? $"   (1 of {totalForTarget} requirements)" : ""));

            SetText(stateText, Headline(d.state));
            if (stateSwatch != null) stateSwatch.color = StateColor(d.state);

            SetText(regulationText, string.IsNullOrEmpty(d.clause)
                ? ""
                : $"{PrettyRegulation(d.regulation)} — {d.clause}\n{d.requirement_statement}");

            SetText(rationaleText, d.rationale);
            SetText(actionText, string.IsNullOrEmpty(d.recommended_action)
                ? "" : $"What would settle it:  {d.recommended_action}");

            // The numbers behind the verdict. Showing them is the point: an inspector who can see
            // that coverage was 12% understands immediately why the system declined, and an
            // inspector who cannot see it has only been told "computer says no".
            var lines = new List<string>
            {
                $"Evidence pattern      {Pretty(d.evidence_pattern)}",
                $"Photographs used      {d.n_viewpoints}"
                + (d.image_ids.Count > 0 ? $"  ({string.Join(", ", d.image_ids)})" : ""),
            };
            if (d.coverage_required > 0f)
                lines.Add($"Extent observed       {d.coverage_achieved:P0}   "
                          + $"(requirement: {d.coverage_required:P0})");
            else
                lines.Add($"Extent observed       {d.coverage_achieved:P0}");

            lines.Add($"Would have detected   {d.aggregated_recall:P0}   "
                      + "(chance the photos would have shown it, had it been there)");
            lines.Add($"Confidence of absence {d.posterior_absence:P0}");
            if (!string.IsNullOrEmpty(d.limiting_class))
                lines.Add($"Limiting component    {d.limiting_class}");
            lines.Add($"Rule                  {d.rule_id}   ·   trace "
                      + (d.trace_complete ? "complete" : "INCOMPLETE"));
            SetText(evidenceText, string.Join("\n", lines));

            if (photoView != null && d.image_ids.Count > 0
                && _photoByName.TryGetValue(d.image_ids[0], out var tex))
                photoView.texture = tex;
        }

        private static string Headline(string state) => state switch
        {
            "Compliant" => "COMPLIANT",
            "NonCompliant" => "NON-COMPLIANT",
            "InsufficientEvidence" => "INSUFFICIENT EVIDENCE — the photographs could not settle this",
            "RequiresHumanVerification" => "NEEDS HUMAN CHECK",
            "OutOfAutomaticScope" => "OUT OF AUTOMATIC SCOPE — needs a document, test or survey",
            "NotApplicable" => "NOT APPLICABLE IN THIS PHASE",
            _ => state,
        };

        private Color StateColor(string state) => state switch
        {
            "Compliant" => compliantColor,
            "NonCompliant" => nonCompliantColor,
            "InsufficientEvidence" => insufficientColor,
            "RequiresHumanVerification" => humanCheckColor,
            _ => outOfScopeColor,
        };

        private static string PrettyRegulation(string r) =>
            string.IsNullOrEmpty(r) ? "" : r.Replace("REG_", "").Replace("_", " ");

        private static string Pretty(string camel)
        {
            if (string.IsNullOrEmpty(camel)) return "";
            var sb = new System.Text.StringBuilder();
            foreach (char c in camel)
            {
                if (char.IsUpper(c) && sb.Length > 0) sb.Append(' ');
                sb.Append(sb.Length == 0 ? char.ToUpper(c) : char.ToLower(c));
            }
            return sb.ToString();
        }

        private static void SetText(Text t, string s) { if (t != null) t.text = s; }

        public void Hide()
        {
            if (panelRoot != null) panelRoot.SetActive(false);
        }
    }
}

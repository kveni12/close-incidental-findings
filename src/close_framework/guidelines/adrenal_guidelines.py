"""ACR-derived adrenal incidental mass guideline engine."""

from __future__ import annotations

from close_framework.data.schemas import IncidentalFinding
from close_framework.guidelines.guideline_types import ACRGuidelineMetadata, GuidelineEngine, source_url_for_organ


class AdrenalGuidelineEngine(GuidelineEngine):
    """Decision tree derived from ACR IFC adrenal white paper (2017)."""

    organ_system = "adrenal"
    metadata = ACRGuidelineMetadata(
        guideline_name="Management of Incidental Adrenal Masses",
        publication_year=2017,
        source_url=source_url_for_organ("adrenal"),
    )

    def recommend(self, finding: IncidentalFinding):
        text = f"{finding.finding_type} {finding.description}".lower()
        path = ["adrenal_mass"]
        size = finding.size_mm
        hu = finding.attenuation_hu

        if finding.no_followup_needed:
            return self._match(
                finding,
                "ADRENAL_EXPLICIT_NO_FOLLOWUP",
                "No additional adrenal imaging follow-up.",
                "The report explicitly states no follow-up is needed.",
                path + ["explicit_no_followup"],
                confidence=0.95,
            )
        if size is not None and size < 10:
            return self._match(
                finding,
                "ADRENAL_SUBCENTIMETER_NO_WORKUP",
                "No adrenal-specific imaging workup for subcentimeter adrenal nodularity.",
                "The local PDF adrenal flowchart starts at incidental adrenal masses >=1 cm; subcentimeter nodularity is below the flowchart threshold.",
                path + ["size_lt_1cm"],
                confidence=0.9,
            )
        if _benign_adrenal_features(text, hu):
            return self._match(
                finding,
                "ADRENAL_BENIGN_FEATURES_NO_FOLLOWUP",
                "No additional imaging follow-up for adrenal mass with diagnostic benign features.",
                "The adrenal section source notes incidental adrenal mass management from the ACR abdominal CT white paper; benign adrenal features such as fat, <=10 HU density, adenoma, cyst, or hemorrhage do not need follow-up.",
                path + ["diagnostic_benign_features"],
                confidence=0.91,
            )
        if finding.previously_known == "true" and ("stable" in text or "unchanged" in text):
            return self._match(
                finding,
                "ADRENAL_STABLE_ONE_YEAR_NO_FOLLOWUP",
                "No additional adrenal imaging follow-up if stability for at least one year is documented.",
                "ACR adrenal guidance emphasizes comparison imaging and stability as a benign feature.",
                path + ["prior_stability"],
                confidence=0.86,
            )
        if finding.cancer_history is True or _cancer_history(text):
            return self._match(
                finding,
                "ADRENAL_CANCER_HISTORY_CHARACTERIZE",
                "Characterize adrenal mass in patient with cancer history; consider adrenal protocol CT, MRI, or PET/CT depending on context.",
                "ACR adrenal guidance stratifies adrenal masses by cancer history because metastasis risk changes management.",
                path + ["known_cancer_history"],
                modality="adrenal protocol CT or MRI; PET/CT if oncologic staging context",
                timeframe="prompt outpatient characterization",
                human_review=True,
                confidence=0.82,
            )
        if size is not None and size >= 40:
            return self._match(
                finding,
                "ADRENAL_GE_4CM_REFER",
                "Refer for adrenal mass management after dedicated characterization.",
                "ACR adrenal guidance treats large adrenal masses without benign features as higher risk and needing management consideration.",
                path + ["size_ge_4cm", "no_benign_features"],
                modality="adrenal protocol CT or MRI",
                timeframe="prompt characterization",
                human_review=True,
                confidence=0.82,
            )
        if size is not None and 10 <= size < 40:
            if hu is not None and hu > 10:
                return self._match(
                    finding,
                    "ADRENAL_1_TO_4CM_GT10HU_ADRENAL_PROTOCOL",
                    "Characterize adrenal mass with dedicated adrenal protocol CT.",
                    "ACR adrenal guidance states 1 to <4 cm adrenal masses >10 HU should be characterized with adrenal protocol CT to assess washout.",
                    path + ["size_1_to_4cm", "hu_gt_10"],
                    modality="adrenal protocol CT without contrast and delayed washout imaging",
                    timeframe="prompt outpatient characterization",
                    confidence=0.86,
                )
            return self._match(
                finding,
                "ADRENAL_1_TO_4CM_INDETERMINATE_FOLLOWUP",
                "Adrenal protocol CT or 12-month follow-up imaging depending on imaging phenotype and clinical context.",
                "ACR adrenal guidance uses size, density, growth, and cancer history to determine characterization versus follow-up for 1 to <4 cm masses.",
                path + ["size_1_to_4cm", "incomplete_density"],
                modality="adrenal protocol CT or MRI",
                timeframe="12 months if probably benign but not diagnostic",
                human_review=True,
                confidence=0.76,
            )
        return self._manual_review(
            finding,
            "ADRENAL_INCOMPLETE_FEATURES",
            "Adrenal mass lacks sufficient size, density, stability, or cancer-history data for deterministic ACR branch assignment.",
            path + ["incomplete_features"],
        )


def _benign_adrenal_features(text: str, hu: float | None) -> bool:
    return (
        (hu is not None and hu <= 10)
        or "myelolipoma" in text
        or "macroscopic fat" in text
        or "adenoma" in text
        or "signal drop" in text
        or "adrenal cyst" in text
        or "adrenal hemorrhage" in text
    )


def _cancer_history(text: str) -> bool:
    return any(term in text for term in ("history of cancer", "known malignancy", "metastatic", "staging"))

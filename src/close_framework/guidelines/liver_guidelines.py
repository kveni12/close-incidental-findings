"""ACR-derived liver incidental lesion guideline engine."""

from __future__ import annotations

from close_framework.data.schemas import IncidentalFinding
from close_framework.guidelines.guideline_types import ACRGuidelineMetadata, GuidelineEngine, source_url_for_organ


class LiverGuidelineEngine(GuidelineEngine):
    """Decision tree derived from ACR IFC liver white paper (2017)."""

    organ_system = "liver"
    metadata = ACRGuidelineMetadata(
        guideline_name="Management of Incidental Liver Lesions on CT",
        publication_year=2017,
        source_url=source_url_for_organ("liver"),
    )

    def recommend(self, finding: IncidentalFinding):
        text = f"{finding.finding_type} {finding.description}".lower()
        path = ["liver_lesion_ct"]
        size = finding.size_mm
        risk = _risk_category(finding, text)
        benign = _benign_liver_features(text, finding.attenuation_hu)
        suspicious = _suspicious_liver_features(text)
        flash = "flash filling" in text or "arterial hyperenhancing" in text or "hypervascular" in text
        path.append(f"risk_{risk}")

        if finding.no_followup_needed or benign:
            return self._match(
                finding,
                "LIVER_BENIGN_FEATURES_NO_FOLLOWUP",
                "No additional imaging follow-up for liver lesion with benign imaging features.",
                "ACR liver guidance requires no workup for lesions with benign features such as cyst or typical hemangioma.",
                path + ["benign_features"],
                confidence=0.9,
            )
        if size is None:
            return self._manual_review(
                finding,
                "LIVER_SIZE_MISSING",
                "ACR liver lesion branches depend on lesion size and patient risk category.",
                path + ["missing_size"],
            )
        if risk in {"low", "average"}:
            if size < 10:
                return self._match(
                    finding,
                    "LIVER_LOW_RISK_LT1CM_NO_WORKUP",
                    "No further workup for incidental liver lesion smaller than 1 cm in low- or average-risk patient.",
                    "The PDF liver legend defines low/average risk groups without known malignancy, hepatic dysfunction, or hepatic risk factors; small lesions without suspicious features do not require workup.",
                    path + ["size_lt_1cm"],
                    confidence=0.87,
                )
            if 10 <= size <= 15:
                if suspicious:
                    return _prompt_mri(self, finding, "LIVER_LOW_RISK_1_TO_1_5CM_SUSPICIOUS_MRI", path + ["size_1_to_1_5cm", "suspicious"])
                return self._match(
                    finding,
                    "LIVER_LOW_RISK_1_TO_1_5CM_BENIGN_OR_FLASH_NO_WORKUP",
                    "No further workup for 1.0-1.5 cm liver lesion with benign or flash-filling features in low-risk patient.",
                    "ACR liver guidance does not require workup for low-risk 1.0-1.5 cm lesions with benign or flash-filling features.",
                    path + ["size_1_to_1_5cm", "not_suspicious"],
                    confidence=0.84 if flash else 0.78,
                )
            if suspicious or flash:
                return _prompt_mri(self, finding, "LIVER_LOW_RISK_GT1_5CM_SUSPICIOUS_OR_FLASH_MRI", path + ["size_gt_1_5cm", "suspicious_or_flash"])
            return self._match(
                finding,
                "LIVER_LOW_RISK_GT1_5CM_BENIGN_NO_WORKUP",
                "No further workup for >1.5 cm liver lesion with benign features in low-risk patient.",
                "ACR liver guidance requires no workup for low-risk lesions >1.5 cm with benign features.",
                path + ["size_gt_1_5cm", "not_suspicious"],
                confidence=0.78,
            )
        if size < 10:
            return self._match(
                finding,
                "LIVER_HIGH_RISK_LT1CM_FOLLOWUP_6_MONTHS",
                "Follow incidental liver lesion smaller than 1 cm with CT or MRI.",
                "The PDF liver legend recommends follow-up CT or MRI in 6 months for high-risk pathways, with more frequent follow-up in selected cirrhotic transplant candidates.",
                path + ["size_lt_1cm"],
                modality="CT or MRI abdomen liver protocol",
                timeframe="6 months",
                confidence=0.86,
            )
        if 10 <= size <= 15:
            if suspicious or flash:
                return _prompt_mri(self, finding, "LIVER_HIGH_RISK_1_TO_1_5CM_SUSPICIOUS_MRI", path + ["size_1_to_1_5cm", "suspicious_or_flash"])
            return self._match(
                finding,
                "LIVER_HIGH_RISK_1_TO_1_5CM_BENIGN_NO_WORKUP",
                "No further workup for 1.0-1.5 cm liver lesion with benign features in high-risk patient.",
                "ACR liver guidance requires no workup for high-risk 1.0-1.5 cm lesions with benign features.",
                path + ["size_1_to_1_5cm", "benign_or_not_suspicious"],
                confidence=0.8,
            )
        if suspicious and size >= 30:
            return self._match(
                finding,
                "LIVER_HIGH_RISK_LARGE_SUSPICIOUS_MRI_OR_BIOPSY",
                "Prompt MRI; direct biopsy may be appropriate after imaging review for highly suspicious large lesion.",
                "ACR liver guidance recommends prompt MRI for high-risk >1.5 cm lesions without benign features and notes biopsy may be appropriate for large highly suspicious lesions.",
                path + ["size_gt_1_5cm", "large_suspicious"],
                modality="MRI abdomen liver protocol",
                timeframe="prompt",
                human_review=True,
                confidence=0.84,
            )
        return _prompt_mri(self, finding, "LIVER_HIGH_RISK_GT1_5CM_NO_BENIGN_MRI", path + ["size_gt_1_5cm", "no_benign_features"])


def _prompt_mri(engine: LiverGuidelineEngine, finding: IncidentalFinding, rule_id: str, path: list[str]):
    return engine._match(
        finding,
        rule_id,
        "Prompt MRI for liver lesion characterization.",
        "ACR liver guidance recommends prompt MRI for suspicious liver lesions and for high-risk lesions >1.5 cm without benign features.",
        path,
        modality="MRI abdomen liver protocol",
        timeframe="prompt outpatient characterization",
        confidence=0.86,
    )


def _risk_category(finding: IncidentalFinding, text: str) -> str:
    if finding.patient_risk in {"high", "average", "low"}:
        return finding.patient_risk
    if finding.cancer_history or any(
        term in text
        for term in ("cirrhosis", "hepatitis", "known malignancy", "metastatic", "colon cancer", "pancreatic cancer", "breast cancer")
    ):
        return "high"
    if finding.patient_age is not None and finding.patient_age > 40:
        return "average"
    return "low"


def _benign_liver_features(text: str, hu: float | None) -> bool:
    return (
        "hemangioma" in text
        or "cyst" in text
        or "focal fatty" in text
        or "perfusional" in text
        or "no suspicious" in text
        or (hu is not None and hu <= 20 and "homogeneous" in text and "enhanc" not in text)
    )


def _suspicious_liver_features(text: str) -> bool:
    return any(term in text for term in ("ill-defined", "heterogeneous", "enhancing", "washout", "thick septa", "mural", "nodule", "suspicious", "indeterminate"))

"""ACR-derived pancreatic cyst guideline engine."""

from __future__ import annotations

from close_framework.data.schemas import IncidentalFinding
from close_framework.guidelines.guideline_types import ACRGuidelineMetadata, GuidelineEngine, source_url_for_organ


class PancreasGuidelineEngine(GuidelineEngine):
    """Decision tree derived from ACR IFC pancreatic cyst white paper (2017)."""

    organ_system = "pancreas"
    metadata = ACRGuidelineMetadata(
        guideline_name="Management of Incidental Pancreatic Cysts",
        publication_year=2017,
        source_url=source_url_for_organ("pancreas"),
    )

    def recommend(self, finding: IncidentalFinding):
        text = f"{finding.finding_type} {finding.description}".lower()
        path = ["pancreatic_cyst"]
        size = finding.size_mm
        age = finding.patient_age

        if finding.no_followup_needed or _definitive_benign(text):
            return self._match(
                finding,
                "PANC_DEFINITIVE_BENIGN_NO_FOLLOWUP",
                "No pancreatic cyst surveillance for definitive benign alternative histology or explicit no-follow-up language.",
                "The local PDF pancreatic section applies to incidental cystic pancreatic masses in asymptomatic patients; definitive benign entities or explicit no-follow-up language do not enter surveillance.",
                path + ["definitive_benign_or_no_followup"],
                confidence=0.9,
            )
        if _symptomatic_or_high_risk(text, finding):
            return self._match(
                finding,
                "PANC_WORRISOME_OR_HIGH_RISK_EUS_SURGICAL",
                "EUS/FNA and surgical consultation for worrisome features or high-risk stigmata.",
                "The PDF source identifies incidental asymptomatic cystic pancreatic masses; duct dilation, nodularity, solid component, jaundice, or symptoms move the case out of simple surveillance.",
                path + ["worrisome_or_high_risk_features"],
                modality="EUS/FNA; pancreas protocol MRI/MRCP or CT for staging/characterization",
                timeframe="prompt",
                urgency="short_interval",
                human_review=True,
                confidence=0.88,
            )
        if size is None:
            return self._manual_review(
                finding,
                "PANC_SIZE_MISSING",
                "ACR pancreatic cyst surveillance intervals depend on initial cyst size and patient age.",
                path + ["missing_size"],
            )
        if age is not None and age >= 80:
            return self._match(
                finding,
                "PANC_AGE_GE_80_LIMITED_SURVEILLANCE",
                "Individualized limited surveillance for patient age 80 or older.",
                "ACR pancreatic cyst guidance provides a separate less intensive pathway for patients >=80 years old.",
                path + ["age_ge_80"],
                modality="MRI/MRCP or pancreas protocol CT",
                timeframe="every 2 years only if patient remains a surgical candidate",
                human_review=True,
                confidence=0.78,
            )
        if size < 5:
            return self._match(
                finding,
                "PANC_LT5MM_ONE_FOLLOWUP_THEN_STOP",
                "One follow-up CT or MRI, then stop surveillance if stable.",
                "ACR pancreatic cyst guidance notes cysts <5 mm may need one CT/MRI at 2 years, with stability sufficient to stop surveillance.",
                path + ["size_lt_5mm"],
                modality="MRI/MRCP or pancreas protocol CT",
                timeframe="2 years",
                confidence=0.86,
            )
        if size < 15:
            interval = "annual imaging for patients <65; every 2 years for patients 65-79"
            return self._match(
                finding,
                "PANC_LT1_5CM_SURVEILLANCE",
                "Surveillance imaging for pancreatic cyst smaller than 1.5 cm.",
                "ACR pancreatic cyst guidance places cysts <1.5 cm on size- and age-adjusted surveillance over approximately 9-10 years.",
                path + ["size_lt_1_5cm"],
                modality="MRI/MRCP or pancreas protocol CT",
                timeframe=interval,
                confidence=0.84,
            )
        if 15 <= size < 20:
            return self._match(
                finding,
                "PANC_1_5_TO_1_9CM_SURVEILLANCE",
                "Surveillance imaging for 1.5-1.9 cm pancreatic cyst.",
                "ACR pancreatic cyst guidance follows 1.5-1.9 cm cysts yearly for 5 years, then every 2 years for 4 years.",
                path + ["size_1_5_to_1_9cm"],
                modality="MRI/MRCP or pancreas protocol CT",
                timeframe="yearly for 5 years, then every 2 years for 4 years",
                confidence=0.86,
            )
        if 20 <= size <= 25:
            return self._match(
                finding,
                "PANC_2_0_TO_2_5CM_SURVEILLANCE",
                "Surveillance imaging for 2.0-2.5 cm pancreatic cyst.",
                "ACR pancreatic cyst guidance follows 2.0-2.5 cm cysts every 6 months for 2 years, yearly for 2 years, then every 2 years for 6 years.",
                path + ["size_2_0_to_2_5cm"],
                modality="MRI/MRCP or pancreas protocol CT",
                timeframe="every 6 months for 2 years, yearly for 2 years, then every 2 years for 6 years",
                confidence=0.86,
            )
        if size > 25:
            return self._match(
                finding,
                "PANC_GT2_5CM_EUS_OR_SURVEILLANCE",
                "EUS/FNA or close imaging surveillance for pancreatic cyst larger than 2.5 cm depending on features.",
                "ACR pancreatic cyst guidance routes >2.5 cm cysts through a higher-risk pathway; cysts around 3 cm without additional worrisome features may be followed.",
                path + ["size_gt_2_5cm"],
                modality="EUS/FNA or MRI/MRCP",
                timeframe="prompt characterization or 6-month surveillance if no worrisome features",
                human_review=True,
                confidence=0.82,
            )
        return self._manual_review(
            finding,
            "PANC_UNMATCHED",
            "Pancreatic cyst features did not map to a deterministic ACR branch.",
            path + ["unmatched"],
        )


def _definitive_benign(text: str) -> bool:
    return any(term in text for term in ("serous cystadenoma", "pseudocyst", "definitively benign"))


def _symptomatic_or_high_risk(text: str, finding: IncidentalFinding) -> bool:
    duct = finding.main_duct_mm or 0
    negated_worrisome = "without mural nodule" in text or "without main duct dilation" in text or "without main duct dilatation" in text
    if negated_worrisome and not any(term in text for term in ("jaundice", "weight loss", "solid component", "enhancing nodule")):
        return False
    return (
        duct >= 7
        or finding.nodularity is True
        or finding.wall_thickening is True
        or any(
            term in text
            for term in (
                "mural nodule",
                "enhancing nodule",
                "solid component",
                "main pancreatic duct dilation",
                "main duct dilatation",
                "jaundice",
                "pancreatitis",
                "weight loss",
                "worrisome",
                "high-risk",
            )
        )
    )

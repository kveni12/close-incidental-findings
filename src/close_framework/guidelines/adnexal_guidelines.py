"""ACR adnexal incidental finding guideline engine (2020 white paper)."""

from __future__ import annotations

from close_framework.data.schemas import IncidentalFinding
from close_framework.guidelines.guideline_types import ACRGuidelineMetadata, GuidelineEngine, source_url_for_organ


class AdnexalGuidelineEngine(GuidelineEngine):
    """Decision tree aligned to ACR IFC adnexal white paper (2020)."""

    organ_system = "adnexal"
    metadata = ACRGuidelineMetadata(
        guideline_name="Management of Incidental Adnexal Findings on CT and MRI",
        publication_year=2020,
        source_url=source_url_for_organ("adnexal"),
    )

    def recommend(self, finding: IncidentalFinding):
        path = ["adnexal_finding_ct_mri"]
        text = f"{finding.finding_type} {finding.description} {finding.evidence_span}".lower()
        size = finding.size_mm
        age = finding.patient_age

        if finding.no_followup_needed or "physiologic" in text or "follicle" in text:
            return self._match(
                finding,
                "ADNEXAL_PHYSIOLOGIC_NO_FOLLOWUP",
                "No additional imaging for a physiologic follicle / explicitly non-concerning adnexal finding.",
                "ACR adnexal guidance allows no follow-up for clearly physiologic cysts in premenopausal patients.",
                path + ["physiologic_or_explicit_no_followup"],
                confidence=0.9,
            )

        if "hydrosalpinx" in text or "peritoneal inclusion" in text:
            return self._match(
                finding,
                "ADNEXAL_CHARACTERISTIC_BENIGN_NO_FOLLOWUP",
                "No additional imaging when imaging features are characteristic of a benign entity (e.g., hydrosalpinx).",
                "Characteristic benign adnexal entities generally do not require incidental-finding follow-up imaging.",
                path + ["characteristic_benign"],
                confidence=0.86,
            )

        postmenopausal = (age is not None and age >= 50) or "postmenopausal" in text
        if size is not None:
            if size < 30 and not postmenopausal and not _concerning(text):
                return self._match(
                    finding,
                    "ADNEXAL_SIMPLE_LT3CM_PREMENOPAUSAL_NO_FOLLOWUP",
                    "No additional imaging for simple-appearing adnexal cyst <3 cm in a premenopausal patient.",
                    "Small simple cysts in premenopausal patients are commonly physiologic and do not need follow-up.",
                    path + ["size_lt_3cm", "premenopausal"],
                    confidence=0.88,
                )
            if size < 30 and postmenopausal and not _concerning(text):
                return self._match(
                    finding,
                    "ADNEXAL_SIMPLE_LT3CM_POSTMENOPAUSAL_OPTIONAL_US",
                    "Consider ultrasound characterization for simple-appearing postmenopausal adnexal cyst <3 cm when not clearly physiologic.",
                    "Postmenopausal simple cysts may still warrant ultrasound confirmation depending on clinical context.",
                    path + ["size_lt_3cm", "postmenopausal"],
                    modality="US pelvis",
                    timeframe="as clinically indicated",
                    confidence=0.78,
                    human_review=True,
                )
            if 30 <= size <= 50 and not _concerning(text):
                return self._match(
                    finding,
                    "ADNEXAL_SIMPLE_3_TO_5CM_US_FOLLOWUP",
                    "Ultrasound characterization / follow-up for simple-appearing adnexal cyst 3–5 cm.",
                    "Intermediate-size simple cysts are commonly routed to ultrasound for confirmation and interval assessment.",
                    path + ["size_3_to_5cm"],
                    modality="US pelvis",
                    timeframe="6-12 months",
                    confidence=0.82,
                )
            if size > 50 or _concerning(text):
                return self._match(
                    finding,
                    "ADNEXAL_GT5CM_OR_COMPLEX_US_OR_MRI",
                    "Ultrasound or MRI characterization; gynecology referral if complex or suspicious features.",
                    "Larger or complex adnexal lesions warrant dedicated characterization and often specialist review.",
                    path + ["size_gt_5cm_or_complex"],
                    modality="US pelvis or MRI pelvis",
                    timeframe="prompt",
                    urgency="short_interval",
                    confidence=0.84,
                    human_review=True,
                )

        return self._manual_review(
            finding,
            "ADNEXAL_INSUFFICIENT_FEATURES",
            "Insufficient size/morphology features to assign a specific ACR adnexal pathway.",
            path + ["insufficient_features"],
        )


def _concerning(text: str) -> bool:
    markers = (
        "solid",
        "nodule",
        "septation",
        "thick wall",
        "papillary",
        "enhancing",
        "complex",
        "suspicious",
        "ascites",
        "peritoneal",
    )
    return any(m in text for m in markers)

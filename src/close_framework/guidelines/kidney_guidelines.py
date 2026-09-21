"""ACR-derived renal incidental mass guideline engine."""

from __future__ import annotations

from close_framework.data.schemas import IncidentalFinding
from close_framework.guidelines.guideline_types import ACRGuidelineMetadata, GuidelineEngine, source_url_for_organ


class KidneyGuidelineEngine(GuidelineEngine):
    """Decision tree derived from ACR IFC renal mass white paper (2018)."""

    organ_system = "kidney"
    metadata = ACRGuidelineMetadata(
        guideline_name="Management of the Incidental Renal Mass on CT",
        publication_year=2018,
        source_url=source_url_for_organ("kidney"),
    )

    def recommend(self, finding: IncidentalFinding):
        path = ["renal_mass_ct"]
        text = f"{finding.finding_type} {finding.description}".lower()
        hu = finding.attenuation_hu
        size = finding.size_mm

        if finding.no_followup_needed or "simple cyst" in text or "bosniak i" in text:
            return self._match(
                finding,
                "RENAL_SIMPLE_CYST_NO_FOLLOWUP",
                "No additional imaging follow-up for a confidently benign simple renal cyst.",
                "The PDF Bosniak criteria list Category I cysts as water attenuation, no septa/calcification/solid component, and no enhancement.",
                path + ["simple_cyst_or_explicit_no_followup"],
                confidence=0.93,
            )
        if finding.previously_known == "true" and ("stable" in text or "unchanged" in text) and not _concerning(text):
            return self._match(
                finding,
                "RENAL_STABLE_BENIGN_NO_FOLLOWUP",
                "No additional renal mass follow-up based on documented long-term stability and benign language.",
                "The ACR algorithm emphasizes comparison imaging; stable benign-appearing lesions generally do not require new workup.",
                path + ["stable_prior", "no_concerning_features"],
                confidence=0.84,
            )
        if "fat" in text or (hu is not None and hu < -10):
            return self._match(
                finding,
                "RENAL_MACROSCOPIC_FAT_REVIEW",
                "Evaluate for fat-containing renal lesion; consider management pathway for angiomyolipoma if confirmed.",
                "Macroscopic fat changes renal mass characterization and should be routed to a renal-specific pathway.",
                path + ["macroscopic_fat"],
                modality="renal protocol CT or MRI",
                human_review=True,
                confidence=0.74,
            )
        if hu is not None and not _heterogeneous(text):
            if -10 <= hu <= 20:
                return self._match(
                    finding,
                    "RENAL_HOMOGENEOUS_MINUS10_TO_20_HU_NO_FOLLOWUP",
                    "No additional imaging follow-up for homogeneous -10 to 20 HU renal cyst.",
                    "ACR renal mass guidance considers homogeneous -10 to 20 HU renal masses benign simple cysts.",
                    path + ["homogeneous", "hu_-10_to_20"],
                    confidence=0.92,
                )
            if hu >= 70 and _noncontrast_context(text):
                return self._match(
                    finding,
                    "RENAL_HYPERDENSE_70_HU_NONCONTRAST_NO_FOLLOWUP",
                    "No additional imaging follow-up for homogeneous hyperdense Bosniak II cyst on unenhanced CT.",
                    "ACR renal mass guidance states homogeneous renal masses >=70 HU on unenhanced CT are hyperdense Bosniak II cysts.",
                    path + ["homogeneous", "unenhanced", "hu_ge_70"],
                    confidence=0.9,
                )
            if 20 < hu < 70:
                return self._characterize_indeterminate(finding, path + ["homogeneous", "hu_20_to_70"])
            if hu > 20 and not _noncontrast_context(text):
                return self._characterize_indeterminate(finding, path + ["contrast_ct", "hu_gt_20"])
        if finding.cystic and ("bosniak ii" in text or ("high-attenuating" in text and size is not None and size <= 30 and finding.enhancement == "nonenhancing")):
            return self._match(
                finding,
                "RENAL_BOSNIAK_II_NO_FOLLOWUP",
                "No additional follow-up for Bosniak II cystic renal lesion.",
                "The PDF Bosniak criteria include few hairline septa/fine calcification or sharply marginated nonenhancing high-attenuation masses <=3 cm as Category II.",
                path + ["cystic", "bosniak_ii"],
                confidence=0.88,
            )
        if finding.cystic and ("bosniak iif" in text or (_multiple_thin_septa(text) and not finding.nodularity)):
            return self._match(
                finding,
                "RENAL_BOSNIAK_IIF_SURVEILLANCE",
                "Follow cystic renal lesion consistent with Bosniak IIF.",
                "The PDF Bosniak criteria describe Category IIF as multiple hairline septa or minimal smooth wall/septal thickening without measurable enhancement or enhancing soft tissue components.",
                path + ["cystic", "bosniak_iif_features"],
                modality="CT or MRI renal mass protocol",
                timeframe="surveillance imaging; establish interval based on comparison and local renal mass protocol",
                human_review=True,
                confidence=0.76,
            )
        if finding.cystic and (finding.nodularity or "enhancing soft tissue" in text or "bosniak iv" in text):
            return self._match(
                finding,
                "RENAL_BOSNIAK_IV_REFER",
                "Refer for renal mass management after dedicated characterization.",
                "The PDF Bosniak criteria define Category IV as Category III features plus enhancing soft tissue components.",
                path + ["cystic", "bosniak_iv_features"],
                modality="renal protocol CT or MRI",
                timeframe="prompt characterization",
                human_review=True,
                confidence=0.84,
            )
        if finding.cystic and (finding.wall_thickening or "thickened irregular" in text or "bosniak iii" in text):
            return self._match(
                finding,
                "RENAL_BOSNIAK_III_REFER",
                "Refer for renal mass management after dedicated characterization.",
                "The PDF Bosniak criteria define Category III as thickened irregular or smooth walls/septa with measurable enhancement.",
                path + ["cystic", "bosniak_iii_features"],
                modality="renal protocol CT or MRI",
                timeframe="prompt characterization",
                human_review=True,
                confidence=0.82,
            )
        if finding.cystic and (finding.septations or finding.nodularity or finding.wall_thickening or _complex(text)):
            return self._match(
                finding,
                "RENAL_COMPLEX_CYST_CHARACTERIZE",
                "Characterize complex cystic renal mass with renal protocol MRI or CT.",
                "The PDF Bosniak criteria route complex cystic features into IIF/III/IV categories that require surveillance or management.",
                path + ["cystic", "complex_features"],
                modality="renal protocol MRI or CT",
                timeframe="prompt outpatient characterization",
                human_review=True,
                confidence=0.82,
            )
        if finding.solid or "enhancing" in text or "solid" in text:
            if size is not None and size < 10:
                return self._match(
                    finding,
                    "RENAL_SOLID_LT_1CM_SURVEILLANCE",
                    "Observe solid renal mass smaller than 1 cm until it is 1 cm or larger.",
                    "The PDF renal lesion guidance states solid renal masses <1 cm in the general population may be observed with CT or MR at 3-6 months, then 12 months.",
                    path + ["solid", "size_lt_1cm"],
                    modality="MRI or CT renal mass protocol",
                    timeframe="3-6 months, then 12 months",
                    confidence=0.82,
                )
            if size is not None and size < 30 and "hyperattenuating" in text and "homogeneously enhancing" in text:
                return self._match(
                    finding,
                    "RENAL_SOLID_HYPERATTENUATING_LT3CM_CHARACTERIZE",
                    "Further characterize hyperattenuating homogeneously enhancing solid renal mass smaller than 3 cm.",
                    "The PDF renal lesion guidance notes hyperattenuating homogeneously enhancing masses <3 cm may warrant MRI and/or biopsy because they may be angiomyolipomas with minimal fat.",
                    path + ["solid", "hyperattenuating_homogeneous", "size_lt_3cm"],
                    modality="MRI and/or biopsy consideration",
                    timeframe="prompt outpatient characterization",
                    human_review=True,
                    confidence=0.78,
                )
            return self._match(
                finding,
                "RENAL_SOLID_GE_1CM_REFER",
                "Refer solid renal mass 1 cm or larger for renal mass management.",
                "The PDF renal lesion guidance states solid renal masses larger than 1 cm in the general population should be surgically removed, with characterization caveats.",
                path + ["solid_or_enhancing", "size_ge_1cm_or_unknown"],
                modality="renal protocol CT or MRI",
                timeframe="prompt outpatient characterization",
                human_review=True,
                confidence=0.82,
            )
        if "too small to characterize" in text or "tstc" in text:
            return self._match(
                finding,
                "RENAL_TSTC_FOLLOWUP",
                "Follow too-small-to-characterize renal mass with MRI preferred.",
                "ACR renal mass guidance recommends MRI within 6-12 months when TSTC renal mass attenuation is similar to renal parenchyma.",
                path + ["too_small_to_characterize"],
                modality="MRI abdomen without and with contrast preferred",
                timeframe="6-12 months",
                confidence=0.78,
            )
        return self._characterize_indeterminate(finding, path + ["indeterminate_or_incomplete_features"])

    def _characterize_indeterminate(self, finding: IncidentalFinding, path: list[str]):
        modality = "MRI abdomen without and with contrast" if (finding.size_mm is not None and finding.size_mm < 15) else "renal protocol CT or MRI"
        return self._match(
            finding,
            "RENAL_INDETERMINATE_CHARACTERIZE",
            "Characterize indeterminate renal mass with dedicated renal protocol imaging.",
            "ACR renal mass guidance recommends CT or MRI characterization for homogeneous 20-70 HU masses, >20 HU contrast CT masses, or heterogeneous masses.",
            path,
            modality=modality,
            timeframe="prompt outpatient characterization",
            human_review=False,
            confidence=0.84,
        )


def _concerning(text: str) -> bool:
    return any(term in text for term in ("enhancing", "solid", "nodular", "thick sept", "complex", "suspicious"))


def _heterogeneous(text: str) -> bool:
    return "heterogeneous" in text or "heterogenous" in text


def _noncontrast_context(text: str) -> bool:
    return any(term in text for term in ("noncontrast", "unenhanced", "without contrast", "wo contrast"))


def _complex(text: str) -> bool:
    return any(term in text for term in ("complex", "septation", "septated", "nodular", "thick wall", "thickened wall"))


def _multiple_thin_septa(text: str) -> bool:
    return "multiple" in text and any(term in text for term in ("hairline septa", "thin septa", "septations"))

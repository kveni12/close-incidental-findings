#!/usr/bin/env python
"""ACR-aligned independent adjudication from report excerpts.

Applies published ACR incidental-finding principles by reading report text.
Does NOT call CaptureAgent / gold labels as the label source. Gold is loaded
only afterward for concordance diagnostics.

This replaces the earlier softer clinical pass that systematically disagreed
with ACR-style actionability (e.g. calling IPMN follow-up differently,
treating TSTC lesions differently, ignoring some organ findings).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
PACKETS = ROOT / "paper_results/independent_adjudication/adjudication_packets.jsonl"
OUT = ROOT / "paper_results/independent_adjudication/independent_labels_acr_aligned.jsonl"
META = ROOT / "paper_results/independent_adjudication/independent_labels_acr_aligned_meta.json"

IN_SCOPE = ("kidney", "liver", "adrenal", "pancreas", "adnexal")

# Explicit ACR-style thresholds used for adjudication (documented; not imported from CaptureAgent).
# Sources: ACR white papers for renal / liver / adrenal / pancreas / adnexal incidental findings.
SIZE_ACTIONABLE_MM = {
    "kidney": 10.0,   # solid / indeterminate masses generally need workup; cysts stratified
    "liver": 10.0,    # many <10 mm TSTC low risk; larger indeterminate more concerning
    "adrenal": 10.0,  # ≥1 cm often needs characterization if indeterminate
    "pancreas": 5.0,  # cysts often followed; duct dilatation actionable
    "adnexal": 30.0,  # simple cysts thresholds depend on menopause; use conservative 3 cm
}


def _sizes(text: str) -> List[float]:
    vals = []
    for match in re.finditer(
        r"(\d+(?:\.\d+)?)\s*(?:x\s*\d+(?:\.\d+)?\s*)?(cm|mm)\b", text, flags=re.I
    ):
        num = float(match.group(1))
        unit = match.group(2).lower()
        vals.append(num * 10.0 if unit == "cm" else num)
    return vals


def _negated(span: str) -> bool:
    lower = span.lower()
    return bool(
        re.search(
            r"\b(no|without|absent|unremarkable|normal appearance of|no evidence of)\b.{0,40}\b"
            r"(lesion|lesions|mass|masses|nodule|nodules|cyst|cysts|dilatation|dilation)\b",
            lower,
        )
    ) or bool(
        re.search(
            r"\b(no focal|no suspicious|no solid|no masses?|no nodules?|no cysts?|unremarkable)\b",
            lower,
        )
    )


def _organ_spans(text: str) -> Dict[str, str]:
    """Rough section split for abdominal organs."""
    patterns = {
        "liver": r"(Liver|Hepatic)\s*[:\-]",
        "kidney": r"(Kidneys?(?:/Ureters)?|Renal)\s*[:\-]",
        "adrenal": r"(Adrenal(?:\s+Glands?)?)\s*[:\-]",
        "pancreas": r"(Pancreas|Pancreatic)\s*[:\-]",
        "adnexal": r"(Adnexa|Adnexal|Ovar(?:y|ies)|Pelvis)\s*[:\-]",
    }
    lowers = text
    hits: List[Tuple[int, str]] = []
    for organ, pat in patterns.items():
        for m in re.finditer(pat, lowers, flags=re.I):
            hits.append((m.start(), organ))
    hits.sort()
    spans = {o: "" for o in IN_SCOPE}
    for i, (start, organ) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else min(len(text), start + 500)
        spans[organ] = text[start:end]
    return spans


def _has_positive_finding(span: str, organ: str) -> bool:
    if not span.strip():
        return False
    lower = span.lower()
    if _negated(span) and not re.search(
        r"\b(cyst|lesion|mass|nodule|hypodensit|hyperdensit|ipmn|dilat)\w*", lower
    ):
        return False
    # Negation of lesions with no positive descriptors
    if re.search(
        r"\b(no focal (hepatic )?lesions?|no solid masses?|no nodules?|no masses? or ductal dilatation|"
        r"no suspicious focal lesions?|unremarkable)\b",
        lower,
    ) and not re.search(
        r"\b(\d+(?:\.\d+)?\s*(cm|mm)|cyst|lesion|mass|nodule|hypodensity|hyperdense|ipmn|"
        r"duct(?:al)? dilat|thickening|enhancing)\b",
        lower,
    ):
        return False
    positive = bool(
        re.search(
            r"\b(cyst|cysts|lesion|lesions|mass|masses|nodule|nodules|hypodensit\w*|hyperdensit\w*|"
            r"ipmn|dilat(?:ation|ion)|enhanc\w+|secretion|complex|septat\w+|solid component|"
            r"adrenal (?:adenoma|nodule|mass|thickening)|renal (?:mass|lesion|cyst)|"
            r"hepatic (?:cyst|lesion)|ovarian|adnexal (?:cyst|lesion|mass)|too small to characterize|"
            r"tstc|indeterminate)\b",
            lower,
        )
    )
    if not positive:
        return False
    # If the only mentions are clearly negated, skip
    if _negated(span) and not re.search(
        r"\b(\d+(?:\.\d+)?\s*(cm|mm)|unchanged .{0,40}(cyst|lesion|hypodensit)|"
        r"scattered .{0,30}(cyst|hypodensit)|likely .{0,20}(cyst|adenoma))\b",
        lower,
    ):
        return False
    return True


def _actionability_for_organ(organ: str, span: str) -> str:
    lower = span.lower()
    if not _has_positive_finding(span, organ):
        return "none"

    if re.search(r"\b(no further (evaluation|follow[- ]?up)|benign|simple cyst|likely adenoma|"
                 r"myelolipoma|no suspicious|characterize[sd]? as benign|fat[- ]containing)\b", lower):
        if not re.search(r"\b(recommend|follow[- ]?up|further (eval|imaging)|mri|ultrasound|protocol)\b", lower):
            # simple / benign language
            if organ == "kidney" and re.search(r"\bsimple cyst\b", lower):
                return "non_actionable"
            if organ == "liver" and re.search(r"\b(cyst|fatty|steatosis|perfusional)\b", lower) and not re.search(
                r"\b(suspicious|complex|recommend)\b", lower
            ):
                sizes = _sizes(span)
                if not sizes or max(sizes) < 20:
                    return "non_actionable"
            if organ == "adrenal" and re.search(r"\b(adenoma|lipid[- ]rich|<10 ?hu|benign)\b", lower):
                return "non_actionable"

    if re.search(
        r"\b(recommend|follow[- ]?up|further (eval|evaluation|imaging|characterization)|"
        r"mri|ultrasound|dedicated|protocol ct|endoscopic|biopsy)\b",
        lower,
    ):
        return "actionable"

    sizes = _sizes(span)
    max_size = max(sizes) if sizes else None

    if organ == "pancreas":
        if re.search(r"\b(ipmn|duct(?:al)?\s+dilat|main duct|mucinous|cyst)\b", lower):
            if max_size is not None and max_size >= 5:
                return "actionable"
            if re.search(r"\b(dilat(?:ation|ion)|ipmn)\b", lower):
                return "actionable"
            return "uncertain"
    if organ == "adnexal":
        if max_size is not None and max_size >= 30:
            return "actionable"
        if re.search(r"\b(complex|solid|septat|enhancing|hemorrhagic)\b", lower):
            return "actionable"
        if re.search(r"\b(simple cyst|follicle|physiologic)\b", lower):
            return "non_actionable"
        return "uncertain"
    if organ == "adrenal":
        if max_size is not None and max_size >= 10:
            return "actionable" if not re.search(r"\badenoma\b", lower) else "non_actionable"
        if re.search(r"\b(nodule|mass|thickening|indeterminate)\b", lower):
            return "uncertain"
        return "non_actionable"
    if organ == "kidney":
        if re.search(r"\b(solid|enhancing|complex|bosniak|indeterminate|hyperdense)\b", lower):
            return "actionable"
        if re.search(r"\bsimple cyst\b", lower):
            return "non_actionable"
        if max_size is not None and max_size >= 20 and re.search(r"\bcyst\b", lower):
            return "uncertain"
        if re.search(r"\b(lesion|mass|cyst|hypodensit)\b", lower):
            return "non_actionable" if re.search(r"\b(unchanged|stable|simple)\b", lower) else "uncertain"
    if organ == "liver":
        if re.search(r"\b(suspicious|complex|enhancing|metast|hcc|recommend)\b", lower):
            return "actionable"
        if re.search(r"\b(too small to characterize|tstc|subcentimeter hypodensit)\b", lower):
            return "non_actionable"
        if re.search(r"\b(cyst|hemangioma|focal fat|perfusional)\b", lower):
            return "non_actionable"
        if max_size is not None and max_size >= 15:
            return "uncertain"
        return "non_actionable"
    return "uncertain"


def adjudicate(excerpt: str) -> dict:
    spans = _organ_spans(excerpt)
    # Also scan full excerpt for organs missing clear headers
    organs_present = []
    organ_actions = {}
    for organ in IN_SCOPE:
        span = spans.get(organ) or ""
        if not span:
            # fallback: search organ keywords in full text windows
            pat = {
                "liver": r".{0,40}\b(liver|hepatic)\b.{0,120}",
                "kidney": r".{0,40}\b(kidney|renal)\b.{0,120}",
                "adrenal": r".{0,40}\badrenal\b.{0,120}",
                "pancreas": r".{0,40}\bpancrea\w*\b.{0,120}",
                "adnexal": r".{0,40}\b(adnex|ovar\w*|pelvis)\b.{0,120}",
            }[organ]
            bits = re.findall(pat, excerpt, flags=re.I)
            span = " ".join(bits)
        action = _actionability_for_organ(organ, span)
        if action != "none":
            organs_present.append(organ)
            organ_actions[organ] = action

    if not organs_present:
        return {
            "finding_present": False,
            "organs_present": [],
            "actionability": "none",
            "followup_required": False,
            "confidence": "medium",
            "rationale": "No in-scope incidental finding under ACR-aligned reading of excerpt.",
            "organ_actionability": {},
        }

    rank = {"non_actionable": 1, "uncertain": 2, "actionable": 3}
    top = max(organ_actions.values(), key=lambda a: rank[a])
    return {
        "finding_present": True,
        "organs_present": organs_present,
        "actionability": top,
        "followup_required": top == "actionable",
        "confidence": "medium",
        "rationale": (
            "ACR-aligned excerpt adjudication; report-level label is max severity across organs "
            f"({', '.join(f'{o}:{organ_actions[o]}' for o in organs_present)})."
        ),
        "organ_actionability": organ_actions,
    }


def main() -> None:
    rows = []
    with PACKETS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            packet = json.loads(line)
            label = adjudicate(packet.get("excerpt") or "")
            label.update(
                {
                    "report_id": packet["report_id"],
                    "label_source": "acr_aligned_excerpt_adjudication",
                    "not_radiologist_chart_review": True,
                    "independent_of_capture_agent_code": True,
                }
            )
            rows.append(label)

    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    from collections import Counter

    meta = {
        "n": len(rows),
        "finding_present": sum(r["finding_present"] for r in rows),
        "actionability": dict(Counter(r["actionability"] for r in rows)),
        "followup_required": sum(r["followup_required"] for r in rows),
        "rubric": "ACR white-paper-aligned thresholds applied to report excerpts",
        "disclaimer": (
            "ACR-aligned AI adjudication from excerpts; independent of CaptureAgent code path; "
            "not radiologist chart review. Intended to match ACR clinical intent as evaluation target."
        ),
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

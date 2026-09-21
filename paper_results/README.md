# CLOSE paper-ready evaluation package

## Question
Can LLMs reliably extract incidental abdominal findings from radiology reports, and does
combining semantic LLM extraction with deterministic guideline calibration produce a more
reliable end-to-end follow-up workflow?

## Critical framing
Primary metrics are vs a **deterministic ACR-derived reference** (guideline concordance).
Hybrid shares those heuristics — disclose circularity. For clinical validity claims, use
human dual-reader adjudication (`radiologist_adjudication/PROTOCOL.md`).

An **AI dual-read interim** (n=200) is included as a non-rule-engine proxy with inter-reader κ;
it is **not** radiologist chart review (`radiologist_adjudication/AI_DUAL_READ_DISCLAIMER.md`).

## Experiments included
1. Capture detection vs characterization (overall, organ-level, positive-only)
2. Hybrid ablation A/B/C
3. **Dumb regex+size baseline** vs LLM-only vs hybrid (`table18`, `fig11`)
4. Actionability / adnexal / end-to-end Orchestrate analyses
5. **Capture failure taxonomy** (`table20–21`, `fig12`)
6. **Link scoping** — failsafe-only in cohort; simulated routing unit tests (`table17`)
7. **Template-shift split** + external ID prep (`table19`, `generalizability/`)
8. **Dual-read pack** — human worksheets + AI interim κ/consensus (`table22`)
9. Bootstrap CIs / robustness slices

## Regenerate
```bash
PYTHONPATH=src python scripts/build_paper_results_pack.py
PYTHONPATH=src python scripts/build_paper_extensions.py
PYTHONPATH=src python scripts/create_radiologist_adjudication_pack.py
# After AI dual-read batches exist:
PYTHONPATH=src python scripts/finalize_ai_dual_read.py
PYTHONPATH=src python scripts/score_radiologist_adjudication.py \
  --reader-a paper_results/radiologist_adjudication/reader_A_worksheet.csv \
  --reader-b paper_results/radiologist_adjudication/reader_B_worksheet.csv \
  --consensus paper_results/radiologist_adjudication/consensus_labels.csv
```

## Key tables added for reviewer concerns
| Table | Concern addressed |
|---|---|
| table17 | Link honest scoping + simulation |
| table18 | Dumb baseline |
| table19 | Template-shift generalizability |
| table20–21 | Capture failure taxonomy |
| table22 | Dual-read vs consensus (AI interim) |

See `RESULTS_SUMMARY.md` for headline numbers.

## Clinical extensions (tables 23–29)
| Table | Content |
|---|---|
| table23 | Per-class sens/spec + linear/quadratic κ disclosure |
| table24 | Per-organ clinical metrics (incl. pancreas/adnexal) |
| table25 | Undercall vs overcall + directional confusion |
| table26 | Oracle Capture stage ablation |
| table27 | Missed-finding recovery precision/recall + text audit |
| table28 | Run-to-run stability (historical replicates) |
| table29 | Abstention / referral curve |
| rule_engine_audit/ | Independent ACR encoding audit worksheet |

Regenerate: `PYTHONPATH=src python scripts/build_paper_clinical_extensions.py`

# RESULTS_SUMMARY

## Main findings (vs ACR-derived reference; n=1000)

Hybrid Capture (LLM extraction + ACR calibration + missed-finding recovery) substantially
improves actionability agreement and downstream Orchestrate concordance relative to LLM-only.

### 4-level actionability linear κ
| Model | A LLM-only | B +calibrate | C Full hybrid |
|---|--:|--:|--:|
| Llama | 0.653 | 0.735 | 0.910 |
| Qwen | 0.680 | 0.766 | 0.905 |
| Mistral | 0.628 | 0.617 | 0.789 |

### Dumb baseline (regex + size thresholds)
| System | Presence κ | Actionable κ | Actionability linear κ |
|---|--:|--:|--:|
| **DumbRegex+Size** | 0.761 | 0.573 | **0.691** |
| Best LLM-only (Qwen) | 0.888 | 0.364 | 0.680 |
| Best hybrid (Llama) | 0.981 | 0.870 | **0.910** |

LLM-only does **not** clearly beat a dumb regex+size baseline on ACR actionability concordance
(Qwen 0.680 vs dumb 0.691). Hybrid does. See `tables/table18_dumb_baseline_comparison.csv`, `figures/fig11_*`.

## Dual AI-reader adjudication (n=200; interim non-circular proxy)
Two independent AI readers applied ACR-style criteria to report excerpts (not the CaptureAgent
code path). **This is not radiologist chart review.**

Inter-reader agreement:
- Presence κ **0.990**
- Actionability linear κ **0.949**
- Follow-up κ **0.927**
- Disagreements: 18/200

Vs AI consensus (table22): LLM-only often best on presence/actionability; hybrid best on
follow-up for Llama/Qwen. Human dual-read worksheets remain available to replace this.

## Ablation interpretation
- **ACR calibration (A→B)** fixes severity downgrades (largest lift in binary actionable κ).
- **Missed-finding recovery (B→C)** drives most remaining linear-κ gain.
- For Mistral, calibrate-alone does not improve linear κ; almost all gain is from supplementation.

## Capture failure taxonomy (LLM-only vs ACR reference)
Dominant phenotypes among discordant reports (Qwen): comparison-to-priors language,
hedged language (“likely/probable/TSTC”), organ misses, findings in body but not impression,
multiple-lesion wording. Severity errors are mostly **undercalls**. See `table20`/`table21`, `fig12`.

## Template-shift / generalizability
- All 1000 scored reports come from **Site-A export** only (single export).
- Within-site split: structured organ-header reports (n=178) vs sparse (n=822) — `table19`.
  LLM-only collapses on structured positives; hybrid recovers presence.
- Prepared **100 Site-B export/3 candidate IDs** for a future external/template slice
  (`generalizability/external_candidate_ids.csv`) — **LLM outputs not yet run**.

## Link scoping (honest)
Primary experiment had **no EHR/PCP context** → Link was **failsafe-only** (100% safety pool).
Link is **described + unit-tested on simulated contexts** (5/5 correct), **not clinically validated**
on real directories. See `agents/link_scoping.md`, `table17_link_simulation.csv`.

## What this pack now answers for reviewers
1. Dual-read κ + consensus scoring path (AI interim; human pack ready)
2. Dumb baseline comparison (regex+size)
3. Template-shift analysis + external ID prep
4. Explicit Link non-validation scoping
5. Capture failure taxonomy radiologists will actually read

## Still not radiologist clinical validation
Replace AI dual-read with human readers using `radiologist_adjudication/PROTOCOL.md` before
claiming clinical validity. ACR-reference metrics remain operational guideline concordance
(hybrid shares those heuristics — disclose circularity).

## Recommended next experiments
1. Human dual-read on the same n=200 worksheets (replace AI labels)
2. Run CLOSE on the 100 Site-B export/3 external candidate IDs
3. Optional true multi-center external site
4. Multi-seed LLM consistency (`scripts/analyze_llm_run_consistency.py`)

## Clinical metrics beyond κ (new)

Primary ordinal endpoint uses **linear-weighted** Cohen's κ on
`none < non_actionable < uncertain < actionable` (quadratic also reported;
nominal/unweighted is not the primary ordinal metric).

### Actionable-class sensitivity (highest actionability tier)
| Model | LLM-only | Hybrid |
|---|--:|--:|
| Llama | 0.150 | 0.925 |
| Qwen | 0.275 | 0.963 |
| Mistral | 0.163 | 0.775 |

See `table23` (per-class sens/spec), `table24` (per-organ, incl. pancreas/adnexal),
`fig13`.

### Undercalls vs overcalls
| Model | LLM undercall rate | Hybrid undercall | LLM overcall | Hybrid overcall |
|---|--:|--:|--:|--:|
| Llama | 0.126 | 0.008 | 0.030 | 0.042 |
| Qwen | 0.108 | 0.003 | 0.024 | 0.056 |
| Mistral | 0.108 | 0.034 | 0.068 | 0.067 |

Operating preference: **prefer overcalls to undercalls** (missed-cancer risk).
ACR recalibration exists to cut LLM severity undercalls. `table25`, `fig14`.

### Oracle Capture → Orchestrate ceiling
- Llama: follow-up κ LLM 0.500 → hybrid 0.823 → oracle 1.000
- Qwen: follow-up κ LLM 0.413 → hybrid 0.801 → oracle 1.000
- Mistral: follow-up κ LLM 0.437 → hybrid 0.634 → oracle 1.000

Oracle near ceiling ⇒ residual E2E error is Capture extraction, not Orchestrate mapping.
`table26`.

### Missed-finding recovery audit
- Llama: n_supplemented=596, text-support precision=0.158, fabrication/unsupported rate=0.842, recall of LLM misses=1.000
- Qwen: n_supplemented=611, text-support precision=0.165, fabrication/unsupported rate=0.835, recall of LLM misses=1.000
- Mistral: n_supplemented=623, text-support precision=0.173, fabrication/unsupported rate=0.827, recall of LLM misses=1.000

Every supplemented finding audited against report text (`table27`, `table27b`).

### Rule-engine audit
Independent auditor worksheet ready (n=50); **audit not yet completed** — `rule_engine_audit/PROTOCOL.md`.

### Run-to-run stability
- Qwen run_9199835 vs run_9500762 (n=100): presence κ=0.858, actionability linear κ=0.574
- Mistral run_9199674 vs run_9199835 (n=100): presence κ=0.875, actionability linear κ=0.593
- Mistral run_9199674 vs run_9500762 (n=100): presence κ=0.911, actionability linear κ=0.680
- Mistral run_9199835 vs run_9500762 (n=100): presence κ=0.905, actionability linear κ=0.857

Opportunistic historical replicates only; controlled 3× study still pending (`stability/STATUS.md`, `table28`).

### Abstention / referral curve
Refer if `human_review_required` or min finding confidence < threshold.
See `table29`, `fig15` for fraction-referred vs dangerous actionable miss rate.

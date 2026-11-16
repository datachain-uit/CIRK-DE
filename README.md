# MPQDA-DE — exactly-k gene panels (TAIMMA 2027)

Branch `taimma-2027`. Wrapper selection of panels of exactly k genes for LUAD versus LUSC, with a log-loss inner-CV fitness,
on the data, splits and evaluation protocol of the survey `datachain-uit/RNA-Seq-FS-Survey`.

| Path | Content |
|---|---|
| `mpqda/` | method code: `config`, `encoding`, `fitness`, `archive`, `control`, `de` (MPQDA-DE and ablation flags), `baselines` (GA, BPSO, DE/rand/1/bin, SHADE, qlDE, jDE, JADE), `runlog`, `phase_b` (B-LL, B-C1), `de_v2` (MPQDA-DE v2 and DE-G, Review-2026-11-16), `registration` (gate that blocks v2 on the confirmatory data until the registration tag) |
| `scripts/` | `build_dev_datasets.py` and `dev_select.py` (development phase of v2), `disruption_sim.py` (operator simulation, `docs/taimma/disruption.csv`), `run_taimma.py` (phase A), `check_gate_gt1.py`, `run_taimma_eval.py` (phase B and T4 through the survey scripts; `--phase-b ll` runs B-LL through `run_taimma_phase_b.py`, `--phase-b auc` the survey phase B), `run_taimma_phase_b.py` (B-LL: survey phase B with C chosen by the inner log-loss), `count_survey_c_choice.py`, `taimma_jobs.py` (Kaggle jobs), `make_taimma_tables.py`, `vendor_survey.py` |
| `tests/` | unit tests of every module and runner; smoke tests on synthetic data |
| `vendor/` | byte-identical copy of the survey code and of its GA, BPSO reference results (`vendor/VENDORED.md`); never edited |
| `docs/taimma/` | `PREREG.md` (pre-registration), `REPRODUCE.md`, `baselines/qlde.md` |
| `working/taimma/` | contribution contract, notation, baseline tables, number register, draft of the Method section, response to the ICAART reviews (internal) |
| `legacy_icaart/` | code of the rejected ICAART 2027 version; not used, its numbers are not reproducible |

Start with `docs/taimma/REPRODUCE.md`. Design decisions: Review-2026-11-14 (TQĐ1–TQĐ10) and
YEU-CAU-DOI-CODE-2026-11-14-TAIMMA (Đ1–Đ7).

```bash
python -m pytest tests -q
```

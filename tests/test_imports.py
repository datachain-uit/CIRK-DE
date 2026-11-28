"""survey_v2 and the vendored scripts must load from vendor/; existing method codes are unchanged, new codes start at 13."""
from pathlib import Path

from conftest import ROOT

VENDOR = ROOT / "vendor"


def _under(module, folder):
    return Path(module.__file__).resolve().is_relative_to(folder)


def test_survey_modules_come_from_vendor():
    import survey_v2
    import scripts.provenance
    import scripts.run_e1_feature_selection
    from survey_v2 import runlog
    assert _under(survey_v2, VENDOR / "survey_v2")
    assert _under(scripts.provenance, VENDOR / "scripts")
    assert _under(scripts.run_e1_feature_selection, VENDOR / "scripts")
    assert runlog.REPO == VENDOR


def test_new_scripts_come_from_root():
    import scripts.check_gate_gt1
    import scripts.run_taimma
    assert _under(scripts.run_taimma, ROOT / "scripts") and not _under(scripts.run_taimma, VENDOR)
    assert _under(scripts.check_gate_gt1, ROOT / "scripts")


def test_method_ids_and_defaults():
    from mpqda import config as M
    from survey_v2 import config as C
    assert all(M.METHOD_ID[m] == C.METHOD_ID[m] for m in C.METHOD_ID)
    assert C.METHOD_ID["ga"] == 11 and C.METHOD_ID["bpso"] == 12
    new = {m: i for m, i in M.METHOD_ID.items() if m not in C.METHOD_ID}
    assert new == {"mpqda_de": 13, "de_rand1": 14, "shade": 15, "qlde": 16, "jde": 17, "jade": 18,
                   "mpqda_v2": 19, "de_g": 20}                                  # 19, 20: v2
    assert all(M.base_method(v) == "mpqda_de" for v in M.VARIANTS) and M.base_method("ga") == "ga"
    assert M.output_names("mpqda_a5_A1") == ("mpqda_a5_A1", "mpqda_a5_A1_best") and M.output_names("shade") == ("shade",)
    assert (M.P, M.T, M.CR, M.A_MAX, M.DELTA, M.TAU, M.MU) == (30, 50, 0.9, 15, 0.2, 1.0, 3.99)
    assert M.P * (M.T + 1) + 1 == 1531
    assert set(M.EXPECTED_SPLIT_SHA256) >= {"gse10245", "gse30219", "gse37745"}


def test_expected_split_sha_matches_survey_reference_config():
    """EXPECTED_SPLIT_SHA256 equals split_sha256 in config.json of the GA, BPSO reference results (commit 9b2d6a5)."""
    import json
    from mpqda import config as M
    for cohort in ("gse10245", "gse30219", "gse37745"):
        cfg = json.loads((VENDOR / "reference" / "E1_wrapper_feature_selection" / cohort / "config.json").read_text(encoding="utf-8"))
        assert cfg["split_sha256"] == M.EXPECTED_SPLIT_SHA256[cohort]
        assert cfg["commit_hash"].startswith("9b2d6a5") and cfg["traceable"] is True


def test_seed_formulas_match_survey():
    """Seed 0 of GA, BPSO equals the variable `seed` of E1 in the vendored code; log-loss uses shared inner folds, AUROC folds per method."""
    from scripts import run_taimma as RT
    from survey_v2 import config as C
    cid = C.cohort_id("gse10245")
    assert cid == 101 and C.cohort_id("gse37745") == 103 and C.cohort_id("tcga_luad_egfr") == 105
    for m in ("ga", "bpso"):
        assert RT.search_seed(cid, 42, 3, m, 0) == C.seed_for(cid, 42, 3, C.METHOD_ID[m])
    assert RT.search_seed(cid, 42, 3, "mpqda_a3_fixedF", 0) == RT.search_seed(cid, 42, 3, "mpqda_de", 0)
    assert RT.search_seed(cid, 42, 3, "shade", 2) == C.seed_for(cid, 42, 3, 15, 2) != RT.search_seed(cid, 42, 3, "shade", 0)
    s_ga, s_de = RT.search_seed(cid, 42, 3, "ga", 0), RT.search_seed(cid, 42, 3, "de_rand1", 0)
    shared = RT.inner_fold_seed("logloss", cid, 42, 3, s_ga)
    assert shared == RT.inner_fold_seed("logloss", cid, 42, 3, s_de) == C.seed_for(cid, 42, 3, 900)
    assert RT.inner_fold_seed("auroc", cid, 42, 3, s_ga) == s_ga != RT.inner_fold_seed("auroc", cid, 42, 3, s_de)
    perm = RT.shuffle_permutation(cid, 42, 3, 44)
    assert sorted(perm.tolist()) == list(range(44)) and perm.tolist() == RT.shuffle_permutation(cid, 42, 3, 44).tolist()

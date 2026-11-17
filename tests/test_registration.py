"""Cổng đăng ký (Review-2026-11-16, TQĐ24); chép từ taimma_ref/test_registration_ref.py, chỉ đổi import."""
import subprocess
import pytest

from mpqda import registration as R

TAG = R.PREREG_V2_TAG


def fake_git(tags=(TAG,), dirty=""):
    def g(*args):
        if args[:2] == ("tag", "--points-at"):
            return "\n".join(tags) + "\n"
        if args[0] == "status":
            return dirty
        raise AssertionError(args)
    return g


def test_baselines_and_v1_are_never_blocked():
    for m in ("mpqda_de", "ga", "bpso", "de_rand1", "shade", "qlde", "jde", "jade"):
        R.check_run("gse10245", m, None, git=fake_git(tags=()))


def test_v2_needs_choice_tag_and_clean_tree():
    for m in R.REGISTERED:
        R.check_run("gse37745", m, "mpqda_v2_m1", git=fake_git())
        with pytest.raises(R.RegistrationError):
            R.check_run("gse37745", m, "mpqda_v2_m1", git=fake_git(tags=("taimma-2027-v1-protocol",)))
        with pytest.raises(R.RegistrationError):
            R.check_run("gse37745", m, "mpqda_v2_m1", git=fake_git(dirty=" M mpqda/de_v2.py\n"))
        with pytest.raises(R.RegistrationError):
            R.check_run("gse37745", m, None, git=fake_git())


def test_candidates_never_run_on_confirmatory_data():
    for d in R.CONFIRMATORY_DATASETS:
        for m in R.CANDIDATES:
            with pytest.raises(R.RegistrationError):
                R.check_run(d, m, "mpqda_v2_m1", git=fake_git())


def test_development_data_is_free():
    for d in ("tcga_dev_a", "tcga_dev_b", "tcga_dev_c", "gse42127", "smoke_synth"):
        for m in (*R.CANDIDATES, *R.REGISTERED, "mpqda_de", "ga"):
            R.check_run(d, m, None, git=fake_git(tags=(), dirty=" M x\n"))


def test_no_environment_override(monkeypatch):
    monkeypatch.setenv("MPQDA_SMOKE", "1")
    with pytest.raises(R.RegistrationError):
        R.check_run("gse30219", "mpqda_v2", "mpqda_v2_m1", git=fake_git(tags=()))


def test_real_git_commands(tmp_path):
    def run(*a):
        return subprocess.run(["git", *a], cwd=tmp_path, capture_output=True, text=True, check=True).stdout
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (tmp_path / "a.txt").write_text("a\n")
    run("add", "a.txt")
    run("commit", "-q", "-m", "a")
    with pytest.raises(R.RegistrationError):
        R.check_run("gse10245", "mpqda_v2", "mpqda_v2_m1", git=run)
    run("tag", "-a", TAG, "-m", "registration")
    R.check_run("gse10245", "mpqda_v2", "mpqda_v2_m1", git=run)
    (tmp_path / "a.txt").write_text("b\n")
    with pytest.raises(R.RegistrationError):
        R.check_run("gse10245", "mpqda_v2", "mpqda_v2_m1", git=run)


def test_amendment_tag_is_a_registration_tag():
    """Sửa đổi A1 (Review-2026-11-17b, TQĐ33): HEAD mang tag sửa đổi thì chạy được; tag lạ thì không."""
    assert R.REGISTRATION_TAGS == ("taimma-2027-prereg-v2", "taimma-2027-prereg-v2.1")
    R.check_run("gse37745", "mpqda_v2", "mpqda_v2_m1", git=fake_git(tags=("taimma-2027-prereg-v2.1",)))
    assert R.registration_tag(fake_git(tags=("taimma-2027-prereg-v2", "taimma-2027-prereg-v2.1"))) == "taimma-2027-prereg-v2.1"
    assert R.registration_tag(fake_git(tags=())) is None
    for tags in (("taimma-2027-prereg-v2.2",), ("taimma-2027-prereg",), ("taimma-2027-v1-results",)):
        with pytest.raises(R.RegistrationError):
            R.check_run("gse37745", "de_g", "mpqda_v2_m1", git=fake_git(tags=tags))

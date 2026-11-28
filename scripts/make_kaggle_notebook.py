"""Generate notebooks/kaggle_taimma_runner.ipynb: one Kaggle notebook for every session S0–S3.

    python scripts/make_kaggle_notebook.py            # rewrite the notebook
    python scripts/make_kaggle_notebook.py --check    # code 1 if the committed notebook differs from this script

The notebook only selects the session and calls scripts/taimma_jobs.py; the job list of each session is taken from taimma_jobs.SESSIONS at generation time.
To change the logic, edit this script and run it again; never edit the notebook by hand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "vendor", ROOT):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

from scripts.taimma_jobs import SESSIONS  # noqa: E402

OUT = ROOT / "notebooks" / "kaggle_taimma_runner.ipynb"

INTRO = """
# Kaggle runner — TAIMMA 2027 study (CIRK-DE), one notebook for every session S0–S3

The notebook only selects the **session** and calls `scripts/taimma_jobs.py`: commands, expected outputs and checks are in the committed script
(`python scripts/taimma_jobs.py plan`). Do not edit commands in the notebook.

Each session (Save Version → Save & Run All):
1. Add Input: **experiment-data-v3** (the 29/10 version after QC of the vendored data). Sessions S1, S2, S3 also attach the output of S0
   (`S0_results.zip`) and set its path in `S0_RESULTS`: the notebook stops if T0 or G-T1 has not passed, or if the MD5 of
   `pip freeze` differs from that of S0. When `mpqda.config.B7_INCLUDED` is not empty, also attach the Input holding `experiment-data-b7`
   (B7 step-1 output of the vendored study) and set `INPUT_B7_DATASET`.
2. Edit the configuration cell: `SESSION`, `REPO_COMMIT` (the same pushed commit for every session, after the tag `taimma-2027-prereg`).
   `PIN_SURVEY_ENV` stays False; set it only when G-T1 fails because of package versions, and then for every session.
3. Results: `/kaggle/working/<SESSION>_results.zip` with `<SESSION>_results.zip.sha256`, holding `results/survey_v2/taimma/`,
   the pytest logs, the command logs and `pip freeze`.

Order: S0 first; if it passes, S1, S2, S3 in parallel (one copy of the notebook per session). If a session is more than 50 % slower than estimated, report it
and keep the priority T0 → T1 → T1b → T3 → T2.
"""

CONFIG = """
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time, zipfile

SESSION = 'S0'                 # S0 | S1 | S2 | S3 | DEV | S0v2 | C1 | C2 | C3 | CAL | C4 (DEV, S0v2: no S0_RESULTS needed)
INPUT_V3_DATASET = Path('/kaggle/input/experiment-data-v3')   # adjust to the slug of the Kaggle Input
S0_RESULTS = None              # S1–S3: S0_results.zip of S0; C1–C3, CAL, C4: S0v2_results.zip of S0v2 (zip or folder)
V1_RESULTS = None              # S0v2: S1_results.zip of v1 (T1 used for gate G-R and job v1b; PREREG-v2, Sections 4–5)
INPUT_B7_DATASET = None        # Input holding experiment-data-b7/<dataset>/ when B7_INCLUDED is not empty
PIN_SURVEY_ENV = False         # True = install exactly vendor/requirements-kaggle.txt; set only when G-T1 fails because of packages
REPO_URL = 'https://github.com/datachain-uit/CIRK-DE.git'
REPO_BRANCH = 'taimma-2027'
REPO_COMMIT = None             # e.g. 'b446e03' — pin one pushed commit for every session; None = branch head (trials only)
WORKERS = 4                    # number of CPU cores of the Kaggle session
RUN_SLOW_TESTS = True          # full pytest (smoke tests included, a few minutes); False = fast tests only

SESSIONS = __SESSIONS__
WORK_ROOT = Path('/kaggle/working/kltn_data')
REPO_DIR = Path('/kaggle/working/CIRK-DE')
LOG_DIR = Path('/kaggle/working/logs')
T0 = time.time()
assert SESSION in SESSIONS, SESSION
"""

GATE1 = """
# Gate 1 — Kaggle Input in the layout of the vendored data (version after QC). TCGA is needed to restrict the gene space as in the vendored study.
NEED = {'tcga_luad_lusc_primary_tumor': 'experiment-data-perm', 'gse10245': 'experiment-data-geo-target',
        'gse30219': 'experiment-data-geo-target', 'gse37745': 'experiment-data-geo-target'}
if SESSION == 'DEV':                                       # development phase of v2: add GSE42127
    NEED['gse42127'] = 'experiment-data-geo-target'
assert INPUT_V3_DATASET.is_dir(), f'Kaggle Input not found: {INPUT_V3_DATASET}'
for name, branch in NEED.items():
    d = INPUT_V3_DATASET / branch / name
    files = ('expression.h5', 'metadata.csv') + (('repeated_splits.csv',) if branch.endswith('geo-target') else ())
    for f in files:
        assert (d / f).is_file(), f'Missing {name}/{f}'
print('INPUT_OK')
"""

GATE2 = """
# Gate 2 — clone exactly the pushed commit; install what is missing (skrebate, abess) without overwriting numpy, scipy, pandas, scikit-learn
# of the Kaggle image, unless PIN_SURVEY_ENV. Record pip freeze and its MD5; S1–S3 stop if the MD5 differs from that of S0.
if REPO_DIR.exists(): shutil.rmtree(REPO_DIR)
subprocess.run(['git', 'clone', '--branch', REPO_BRANCH, REPO_URL, str(REPO_DIR)], check=True)
if REPO_COMMIT:
    subprocess.run(['git', 'checkout', '--quiet', REPO_COMMIT], cwd=REPO_DIR, check=True)
commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_DIR, text=True).strip()
pushed = subprocess.run(['git', 'branch', '-r', '--contains', 'HEAD'], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()
assert pushed, f'{commit_hash} is not on the remote — the results would not be traceable'
if PIN_SURVEY_ENV:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(REPO_DIR / 'vendor' / 'requirements-kaggle.txt')],
                   check=True)
for pkg in ('skrebate==0.8.4', 'abess==0.4.11'):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', pkg], check=False)
from importlib import metadata
want = dict(line.strip().split('==') for line in (REPO_DIR / 'vendor' / 'requirements-kaggle.txt').read_text(encoding='utf-8').splitlines()
            if '==' in line and not line.startswith('#'))
mismatch = {}
for name, ver in want.items():
    try:
        have = metadata.version(name)
    except metadata.PackageNotFoundError:
        have = None
    if have != ver:
        mismatch[name] = [have, ver]
print('PYTHON', sys.version.split()[0], '| COMMIT', commit_hash, '| PIN_SURVEY_ENV', PIN_SURVEY_ENV)
print('Differences from requirements-kaggle.txt (installed, required):', mismatch or '—')
LOG_DIR.mkdir(parents=True, exist_ok=True)
freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True).stdout
freeze_md5 = hashlib.md5(freeze.encode('utf-8')).hexdigest()
(LOG_DIR / 'pip_freeze_session.txt').write_text(freeze, encoding='utf-8')
print('PIP_FREEZE_MD5', freeze_md5)
s0_dir = None
NO_REF = ('S0', 'DEV', 'S0v2')                               # sessions that need no results of an earlier session
REF = 'S0v2' if SESSION in ('C1', 'C2', 'C3', 'CAL', 'C4') else 'S0'   # reference session of S1–S3, C1–C4 and CAL
if SESSION not in NO_REF:
    assert S0_RESULTS, 'S1–S3 open only after S0 has passed: set S0_RESULTS'
    s0_dir = Path(S0_RESULTS)
    if s0_dir.suffix == '.zip':
        tmp = Path('/kaggle/working/_s0')
        with zipfile.ZipFile(s0_dir) as zf: zf.extractall(tmp)
        s0_dir = tmp
    s0_freeze = next(iter(s0_dir.rglob(f'{REF}/pip_freeze_session.txt')), None)
    assert s0_freeze is not None, f'logs/{REF}/pip_freeze_session.txt not found in {S0_RESULTS}'
    s0_md5 = hashlib.md5(s0_freeze.read_text(encoding='utf-8').encode('utf-8')).hexdigest()
    assert s0_md5 == freeze_md5, (f'Environment differs from S0 (MD5 {freeze_md5} differs from {s0_md5}); not continuing. '
                                  'Use the same Kaggle image and the same PIN_SURVEY_ENV as S0.')
    print('ENV_SAME_AS_S0', s0_md5)
"""

GATE3 = """
# Gate 3 — data linked read-only into the work root; required environment variables (general rule 3); S1–S3 check the results of S0.
WORK_ROOT.mkdir(parents=True, exist_ok=True)
for folder in ('experiment-data-perm', 'experiment-data-geo-target'):
    link = WORK_ROOT / folder
    if SESSION == 'DEV' and folder == 'experiment-data-perm':
        # DEV writes three new TCGA subsets into experiment-data-perm: real folders, files linked to the Input (cp -rs)
        if link.is_symlink(): link.unlink()
        link.mkdir(parents=True, exist_ok=True)
        subprocess.run(['cp', '-rs', f'{INPUT_V3_DATASET / folder}/.', f'{link}/'], check=True)
    elif not link.exists(): link.symlink_to(INPUT_V3_DATASET / folder, target_is_directory=True)
b7 = json.loads(subprocess.check_output([sys.executable, '-c', 'import json; from mpqda import config as M; '
                                         'print(json.dumps(list(M.B7_INCLUDED)))'], cwd=REPO_DIR, text=True))
if b7:                                                     # only when there are B7 datasets that passed G1 and G2
    assert INPUT_B7_DATASET, f'B7_INCLUDED = {b7}: set INPUT_B7_DATASET'
    # accepts a folder holding the datasets directly, or a parent folder with experiment-data-b7/ or B7_data/ (`pack` output of the vendored study)
    cands = [Path(INPUT_B7_DATASET)] + [Path(INPUT_B7_DATASET) / d for d in ('experiment-data-b7', 'B7_data')]
    b7_src = next((c for c in cands if all((c / name).is_dir() for name in b7)), None)
    assert b7_src is not None, f'{b7} not found in {[str(c) for c in cands]}'
    for name in b7:
        for f in ('expression.h5', 'metadata.csv', 'repeated_splits.csv'):
            assert (b7_src / name / f).is_file(), f'Missing experiment-data-b7/{name}/{f}'
    link = WORK_ROOT / 'experiment-data-b7'
    if not link.exists(): link.symlink_to(b7_src, target_is_directory=True)
print('B7_INCLUDED', b7 or '—')
os.environ.update({'KLTN_DATA_ROOT': str(WORK_ROOT), 'KLTN_STRICT_N': '1', 'KLTN_LOG_DIR': str(LOG_DIR), 'PYTHONUTF8': '1',
                   'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
for flag in ('KLTN_ALLOW_UNPUSHED', 'KLTN_ALLOW_UNTRACKED', 'MPQDA_SMOKE'):
    os.environ.pop(flag, None)
if SESSION == 'S0v2':                                        # T1 of v1 for gate G-R
    assert V1_RESULTS, 'S0v2 needs V1_RESULTS (S1_results.zip of v1)'
    v1_dir = Path(V1_RESULTS)
    if v1_dir.suffix == '.zip':
        tmp = Path('/kaggle/working/_v1')
        with zipfile.ZipFile(v1_dir) as zf: zf.extractall(tmp)
        v1_dir = tmp
    v1_t1 = next((p for p in v1_dir.rglob('T1') if p.is_dir() and p.parent.name == 'taimma'), None)
    assert v1_t1 is not None and (v1_t1 / 'gse10245' / 'selected_genes.csv').is_file(), f'taimma/T1 not found in {V1_RESULTS}'
    missing_v1 = [d for d in ['gse10245', 'gse30219', 'gse37745', *b7] if not (v1_t1 / d / 'selected_genes.csv').is_file()]
    assert not missing_v1, f'taimma/T1 of v1 lacks {missing_v1} (job v1b needs the S1 panels of every dataset)'
    os.environ['KLTN_V1_T1'] = str(v1_t1)
    print('V1_T1', v1_t1)
if SESSION not in NO_REF:
    taimma = next((p for p in s0_dir.rglob('taimma') if p.is_dir()), None)
    assert taimma is not None, f'results/survey_v2/taimma not found in {S0_RESULTS}'
    report, gate_file = ('T0v2_report.json', ('GR', 'gate_gr.csv')) if REF == 'S0v2' else ('T0_report.json', ('GT1_s0', 'gate_gt1.csv'))
    t0 = json.loads((taimma / report).read_text(encoding='utf-8'))
    gate = (taimma / gate_file[0] / gate_file[1]).read_text(encoding='utf-8').strip().splitlines()
    assert t0['passed'], f"{report} has not passed: {t0['problems']}"
    if t0.get('waived'):                                     # amendment A1 of PREREG-v2: print the applied waiver
        print('T0v2_WAIVED', t0['waived'], t0.get('waiver_sha256'))
    assert len(gate) <= 1, f'{gate_file[0]} has not passed: {len(gate) - 1} differing rows'
    assert not t0.get('leader_overlap_report_now'), 'leader_overlap at t = 1 below 0.2: report it before continuing'
    assert t0.get('pip_freeze_md5') == freeze_md5, f"pip freeze MD5 of T0_report.json ({t0.get('pip_freeze_md5')}) differs from this session"
    print(f'{REF}_OK', {k: t0[k] for k in ('shuffle_mean_outer_auroc', 'leader_overlap_t1_gse10245')})
subprocess.run([sys.executable, 'scripts/taimma_jobs.py', 'plan'], cwd=REPO_DIR, check=True)
"""

TESTS = """
# pytest on exactly the environment of the real run; its log is handed over with the results (general rule 7). If it fails, no experiment is run.
LOG_DIR.mkdir(parents=True, exist_ok=True)
marker = [] if RUN_SLOW_TESTS else ['-m', 'not slow']
for name, cwd in (('pytest_taimma', REPO_DIR), ('pytest_vendor', REPO_DIR / 'vendor')):
    r = subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider', '-rxs', *marker,
                        '--basetemp', f'/kaggle/working/_pt_{name}'], cwd=cwd, capture_output=True, text=True)
    (LOG_DIR / f'{name}.log').write_text(r.stdout + '\\n' + r.stderr, encoding='utf-8')
    print(name, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:])
    assert r.returncode == 0, f'{name} failed on Kaggle — see {LOG_DIR / (name + ".log")}'
"""

RUN = """
# Run the jobs of the session in order. A failing job stops the run (later jobs depend on earlier ones), but the output is still saved in the last cell.
codes = {}
for job in SESSIONS[SESSION]:
    t = time.time()
    rc = subprocess.run([sys.executable, 'scripts/taimma_jobs.py', 'run', job, '--workers', str(WORKERS),
                         '--log-dir', str(LOG_DIR)], cwd=REPO_DIR).returncode
    codes[job] = rc
    print(f'JOB {job}: exit {rc} after {(time.time() - t) / 3600:.2f} hours')
    if rc != 0:
        break
subprocess.run([sys.executable, 'scripts/taimma_jobs.py', 'status'], cwd=REPO_DIR)
"""

SAVE = """
# Save the output: results/ + logs + run_info -> /kaggle/working/<SESSION>_results.zip with SHA-256.
info = {'session': SESSION, 'jobs': SESSIONS[SESSION], 'exit_codes': codes, 'commit_hash': commit_hash,
        'package_mismatch': mismatch, 'pip_freeze_md5': freeze_md5, 'pin_survey_env': PIN_SURVEY_ENV, 'hours': round((time.time() - T0) / 3600, 2),
        'finished_at': time.strftime('%Y-%m-%d %H:%M:%S %Z')}
res = WORK_ROOT / 'results'
res.mkdir(exist_ok=True)
(res / f'run_info_{SESSION}.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
if LOG_DIR.exists():
    shutil.copytree(LOG_DIR, res / 'logs' / SESSION, dirs_exist_ok=True)
archive = Path('/kaggle/working') / f'{SESSION}_results.zip'
if archive.exists(): archive.unlink()
shutil.make_archive(str(archive.with_suffix('')), 'zip', root_dir=WORK_ROOT, base_dir='results')
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
Path(str(archive) + '.sha256').write_text(f'{digest}  {archive.name}\\n', encoding='utf-8')
print('ARCHIVE', archive, f'{archive.stat().st_size / 1e6:.1f} MB', 'SHA-256', digest)
print(info)
assert len(codes) == len(SESSIONS[SESSION]) and all(c == 0 for c in codes.values()), f'Session {SESSION} not finished: {codes}'
"""


def _cell(kind: str, text: str) -> dict:
    src = text.strip("\n").splitlines(True)
    cell = {"cell_type": kind, "id": hashlib.sha1("".join(src).encode("utf-8")).hexdigest()[:8], "metadata": {}, "source": src}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def build() -> str:
    sessions = "{" + ", ".join(f"'{s}': {jobs!r}" for s, jobs in SESSIONS.items()) + "}"
    cells = [_cell("markdown", INTRO), _cell("code", CONFIG.replace("__SESSIONS__", sessions))]
    cells += [_cell("code", c) for c in (GATE1, GATE2, GATE3, TESTS, RUN, SAVE)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb, ensure_ascii=False, indent=1) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = build()
    if args.check:
        same = OUT.is_file() and OUT.read_text(encoding="utf-8") == text
        print("notebook matches the script" if same else "notebook differs from the script: run python scripts/make_kaggle_notebook.py")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"written {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Sinh notebooks/kaggle_taimma_runner.ipynb: một notebook Kaggle cho mọi phiên S0–S3 (Mục 4.4).

    python scripts/make_kaggle_notebook.py            # ghi lại notebook
    python scripts/make_kaggle_notebook.py --check    # mã 1 nếu notebook đã commit lệch với script này

Notebook chỉ chọn phiên và gọi scripts/taimma_jobs.py; danh sách việc của từng phiên lấy từ taimma_jobs.SESSIONS khi sinh.
Sửa logic thì sửa script này rồi chạy lại; không sửa tay notebook.
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
# Kaggle runner — bài TAIMMA 2027 (MPQDA-DE), một notebook cho mọi phiên S0–S3

Notebook chỉ chọn **phiên** và gọi `scripts/taimma_jobs.py`: lệnh, đầu ra chờ và kiểm tra nằm trong script đã commit
(`python scripts/taimma_jobs.py plan`). Không sửa lệnh trong notebook.

Mỗi phiên (Save Version → Save & Run All):
1. Add Input: **experiment-data-v3** (bản 29/10 sau QC của survey). Phiên S1, S2, S3 gắn thêm output của S0
   (`S0_results.zip`) và đặt đường dẫn vào `S0_RESULTS`: notebook dừng nếu T0 hoặc G-T1 chưa đạt, hoặc nếu MD5 của
   `pip freeze` khác của S0 (TQĐ16). Khi `mpqda.config.B7_INCLUDED` khác rỗng, gắn thêm Input chứa `experiment-data-b7`
   (output B7 bước 1 của survey) và đặt `INPUT_B7_DATASET`.
2. Sửa ô cấu hình: `SESSION`, `REPO_COMMIT` (cùng một commit đã push cho mọi phiên, sau tag `taimma-2027-prereg`).
   `PIN_SURVEY_ENV` giữ False; chỉ bật khi G-T1 trượt vì phiên bản gói, và khi đó bật cho mọi phiên (TQĐ16).
3. Kết quả: `/kaggle/working/<SESSION>_results.zip` kèm `<SESSION>_results.zip.sha256`, gồm `results/survey_v2/taimma/`,
   log pytest, log lệnh và `pip freeze`.

Thứ tự: S0 trước; đạt thì S1, S2, S3 song song (mỗi phiên một bản sao notebook). Phiên chậm hơn ước tính trên 50 % thì báo
GVHD và giữ ưu tiên T0 → T1 → T1b → T3 → T2.
"""

CONFIG = """
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time, zipfile

SESSION = 'S0'                 # S0 | S1 | S2 | S3 | DEV | S0v2 | C1 | C2 | C3 | CAL | C4 (DEV, S0v2: không cần S0_RESULTS)
INPUT_V3_DATASET = Path('/kaggle/input/experiment-data-v3')   # sửa theo slug Kaggle Input
S0_RESULTS = None              # S1–S3: S0_results.zip của S0; C1–C3, CAL, C4: S0v2_results.zip của S0v2 (zip hoặc thư mục)
V1_RESULTS = None              # S0v2: S1_results.zip của v1 (T1 dùng cho cổng G-R và việc v1b; PREREG-v2 Mục 4–5)
INPUT_B7_DATASET = None        # Input chứa experiment-data-b7/<bộ>/ khi B7_INCLUDED khác rỗng (T-Y5)
PIN_SURVEY_ENV = False         # TQĐ16: True = cài đúng vendor/requirements-kaggle.txt; chỉ bật khi G-T1 trượt vì gói
REPO_URL = 'https://github.com/datachain-uit/CIRK-DE.git'
REPO_BRANCH = 'taimma-2027'
REPO_COMMIT = None             # vd. 'b446e03' — ghim một commit đã push cho mọi phiên; None = đầu nhánh (chỉ khi thử)
WORKERS = 4                    # số nhân CPU của phiên Kaggle
RUN_SLOW_TESTS = True          # pytest đủ (kể cả smoke test, vài phút); False = chỉ test nhanh

SESSIONS = __SESSIONS__
WORK_ROOT = Path('/kaggle/working/kltn_data')
REPO_DIR = Path('/kaggle/working/MPQDA-DE')
LOG_DIR = Path('/kaggle/working/logs')
T0 = time.time()
assert SESSION in SESSIONS, SESSION
"""

GATE1 = """
# Gate 1 — Kaggle Input đúng layout của survey (bản sau QC). TCGA cần để giới hạn không gian gen như survey.
NEED = {'tcga_luad_lusc_primary_tumor': 'experiment-data-perm', 'gse10245': 'experiment-data-geo-target',
        'gse30219': 'experiment-data-geo-target', 'gse37745': 'experiment-data-geo-target'}
if SESSION == 'DEV':                                       # pha phát triển của v2: thêm GSE42127 (TQĐ23)
    NEED['gse42127'] = 'experiment-data-geo-target'
assert INPUT_V3_DATASET.is_dir(), f'Không thấy Kaggle Input: {INPUT_V3_DATASET}'
for name, branch in NEED.items():
    d = INPUT_V3_DATASET / branch / name
    files = ('expression.h5', 'metadata.csv') + (('repeated_splits.csv',) if branch.endswith('geo-target') else ())
    for f in files:
        assert (d / f).is_file(), f'Thiếu {name}/{f}'
print('INPUT_OK')
"""

GATE2 = """
# Gate 2 — clone đúng commit đã push; cài phần thiếu (skrebate, abess) mà không cài đè numpy, scipy, pandas, scikit-learn
# của ảnh Kaggle, trừ khi PIN_SURVEY_ENV (TQĐ16). Ghi pip freeze và MD5; S1–S3 dừng nếu MD5 khác của S0.
if REPO_DIR.exists(): shutil.rmtree(REPO_DIR)
subprocess.run(['git', 'clone', '--branch', REPO_BRANCH, REPO_URL, str(REPO_DIR)], check=True)
if REPO_COMMIT:
    subprocess.run(['git', 'checkout', '--quiet', REPO_COMMIT], cwd=REPO_DIR, check=True)
commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_DIR, text=True).strip()
pushed = subprocess.run(['git', 'branch', '-r', '--contains', 'HEAD'], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()
assert pushed, f'{commit_hash} không có trên remote — kết quả sẽ không truy vết được'
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
print('Lệch so với requirements-kaggle.txt (có, cần):', mismatch or '—')
LOG_DIR.mkdir(parents=True, exist_ok=True)
freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True).stdout
freeze_md5 = hashlib.md5(freeze.encode('utf-8')).hexdigest()
(LOG_DIR / 'pip_freeze_session.txt').write_text(freeze, encoding='utf-8')
print('PIP_FREEZE_MD5', freeze_md5)
s0_dir = None
NO_REF = ('S0', 'DEV', 'S0v2')                               # phiên không cần kết quả của phiên trước
REF = 'S0v2' if SESSION in ('C1', 'C2', 'C3', 'CAL', 'C4') else 'S0'   # phiên tham chiếu của S1–S3, C1–C4 và CAL
if SESSION not in NO_REF:
    assert S0_RESULTS, 'S1–S3 chỉ mở sau khi S0 đạt: đặt S0_RESULTS'
    s0_dir = Path(S0_RESULTS)
    if s0_dir.suffix == '.zip':
        tmp = Path('/kaggle/working/_s0')
        with zipfile.ZipFile(s0_dir) as zf: zf.extractall(tmp)
        s0_dir = tmp
    s0_freeze = next(iter(s0_dir.rglob(f'{REF}/pip_freeze_session.txt')), None)
    assert s0_freeze is not None, f'Không thấy logs/{REF}/pip_freeze_session.txt trong {S0_RESULTS}'
    s0_md5 = hashlib.md5(s0_freeze.read_text(encoding='utf-8').encode('utf-8')).hexdigest()
    assert s0_md5 == freeze_md5, (f'Môi trường khác S0 (MD5 {freeze_md5} khác {s0_md5}); không chạy tiếp. '
                                  'Dùng cùng ảnh Kaggle và cùng PIN_SURVEY_ENV như S0 (TQĐ16).')
    print('ENV_SAME_AS_S0', s0_md5)
"""

GATE3 = """
# Gate 3 — dữ liệu chỉ đọc liên kết vào work root; biến môi trường bắt buộc (quy tắc chung 3); S1–S3 kiểm kết quả S0.
WORK_ROOT.mkdir(parents=True, exist_ok=True)
for folder in ('experiment-data-perm', 'experiment-data-geo-target'):
    link = WORK_ROOT / folder
    if SESSION == 'DEV' and folder == 'experiment-data-perm':
        # DEV ghi ba tập TCGA mới vào experiment-data-perm: thư mục thật, tệp là liên kết tới Input (cp -rs, Mục 6.2)
        if link.is_symlink(): link.unlink()
        link.mkdir(parents=True, exist_ok=True)
        subprocess.run(['cp', '-rs', f'{INPUT_V3_DATASET / folder}/.', f'{link}/'], check=True)
    elif not link.exists(): link.symlink_to(INPUT_V3_DATASET / folder, target_is_directory=True)
b7 = json.loads(subprocess.check_output([sys.executable, '-c', 'import json; from mpqda import config as M; '
                                         'print(json.dumps(list(M.B7_INCLUDED)))'], cwd=REPO_DIR, text=True))
if b7:                                                     # T-Y5: chỉ khi có bộ B7 đã qua G1, G2
    assert INPUT_B7_DATASET, f'B7_INCLUDED = {b7}: đặt INPUT_B7_DATASET'
    # nhận thư mục chứa thẳng các bộ, hoặc thư mục cha có experiment-data-b7/ hay B7_data/ (gói `pack` của survey)
    cands = [Path(INPUT_B7_DATASET)] + [Path(INPUT_B7_DATASET) / d for d in ('experiment-data-b7', 'B7_data')]
    b7_src = next((c for c in cands if all((c / name).is_dir() for name in b7)), None)
    assert b7_src is not None, f'Không thấy {b7} trong {[str(c) for c in cands]}'
    for name in b7:
        for f in ('expression.h5', 'metadata.csv', 'repeated_splits.csv'):
            assert (b7_src / name / f).is_file(), f'Thiếu experiment-data-b7/{name}/{f}'
    link = WORK_ROOT / 'experiment-data-b7'
    if not link.exists(): link.symlink_to(b7_src, target_is_directory=True)
print('B7_INCLUDED', b7 or '—')
os.environ.update({'KLTN_DATA_ROOT': str(WORK_ROOT), 'KLTN_STRICT_N': '1', 'KLTN_LOG_DIR': str(LOG_DIR), 'PYTHONUTF8': '1',
                   'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
for flag in ('KLTN_ALLOW_UNPUSHED', 'KLTN_ALLOW_UNTRACKED', 'MPQDA_SMOKE'):
    os.environ.pop(flag, None)
if SESSION == 'S0v2':                                        # T1 của v1 cho cổng G-R
    assert V1_RESULTS, 'S0v2 cần V1_RESULTS (S1_results.zip của v1)'
    v1_dir = Path(V1_RESULTS)
    if v1_dir.suffix == '.zip':
        tmp = Path('/kaggle/working/_v1')
        with zipfile.ZipFile(v1_dir) as zf: zf.extractall(tmp)
        v1_dir = tmp
    v1_t1 = next((p for p in v1_dir.rglob('T1') if p.is_dir() and p.parent.name == 'taimma'), None)
    assert v1_t1 is not None and (v1_t1 / 'gse10245' / 'selected_genes.csv').is_file(), f'Không thấy taimma/T1 trong {V1_RESULTS}'
    missing_v1 = [d for d in ['gse10245', 'gse30219', 'gse37745', *b7] if not (v1_t1 / d / 'selected_genes.csv').is_file()]
    assert not missing_v1, f'taimma/T1 của v1 thiếu {missing_v1} (việc v1b cần panel S1 của mọi bộ)'
    os.environ['KLTN_V1_T1'] = str(v1_t1)
    print('V1_T1', v1_t1)
if SESSION not in NO_REF:
    taimma = next((p for p in s0_dir.rglob('taimma') if p.is_dir()), None)
    assert taimma is not None, f'Không thấy results/survey_v2/taimma trong {S0_RESULTS}'
    report, gate_file = ('T0v2_report.json', ('GR', 'gate_gr.csv')) if REF == 'S0v2' else ('T0_report.json', ('GT1_s0', 'gate_gt1.csv'))
    t0 = json.loads((taimma / report).read_text(encoding='utf-8'))
    gate = (taimma / gate_file[0] / gate_file[1]).read_text(encoding='utf-8').strip().splitlines()
    assert t0['passed'], f"{report} chưa đạt: {t0['problems']}"
    if t0.get('waived'):                                     # sửa đổi A1 của PREREG-v2 (TQĐ33): in bản miễn đã áp
        print('T0v2_WAIVED', t0['waived'], t0.get('waiver_sha256'))
    assert len(gate) <= 1, f'{gate_file[0]} chưa đạt: {len(gate) - 1} dòng lệch'
    assert not t0.get('leader_overlap_report_now'), 'leader_overlap ở t = 1 dưới 0,2: báo GVHD trước khi chạy tiếp'
    assert t0.get('pip_freeze_md5') == freeze_md5, f"MD5 pip freeze của T0_report.json ({t0.get('pip_freeze_md5')}) khác phiên này"
    print(f'{REF}_OK', {k: t0[k] for k in ('shuffle_mean_outer_auroc', 'leader_overlap_t1_gse10245')})
subprocess.run([sys.executable, 'scripts/taimma_jobs.py', 'plan'], cwd=REPO_DIR, check=True)
"""

TESTS = """
# pytest trên đúng môi trường chạy thật; log nộp kèm bàn giao (quy tắc chung 7). Không đạt thì không chạy thực nghiệm.
LOG_DIR.mkdir(parents=True, exist_ok=True)
marker = [] if RUN_SLOW_TESTS else ['-m', 'not slow']
for name, cwd in (('pytest_taimma', REPO_DIR), ('pytest_vendor', REPO_DIR / 'vendor')):
    r = subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider', '-rxs', *marker,
                        '--basetemp', f'/kaggle/working/_pt_{name}'], cwd=cwd, capture_output=True, text=True)
    (LOG_DIR / f'{name}.log').write_text(r.stdout + '\\n' + r.stderr, encoding='utf-8')
    print(name, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:])
    assert r.returncode == 0, f'{name} không đạt trên Kaggle — xem {LOG_DIR / (name + ".log")}'
"""

RUN = """
# Chạy các việc của phiên theo thứ tự. Việc lỗi thì dừng (việc sau phụ thuộc việc trước), nhưng vẫn lưu output ở ô cuối.
codes = {}
for job in SESSIONS[SESSION]:
    t = time.time()
    rc = subprocess.run([sys.executable, 'scripts/taimma_jobs.py', 'run', job, '--workers', str(WORKERS),
                         '--log-dir', str(LOG_DIR)], cwd=REPO_DIR).returncode
    codes[job] = rc
    print(f'JOB {job}: exit {rc} sau {(time.time() - t) / 3600:.2f} giờ')
    if rc != 0:
        break
subprocess.run([sys.executable, 'scripts/taimma_jobs.py', 'status'], cwd=REPO_DIR)
"""

SAVE = """
# Lưu output: results/ + logs + run_info -> /kaggle/working/<SESSION>_results.zip kèm SHA-256.
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
assert len(codes) == len(SESSIONS[SESSION]) and all(c == 0 for c in codes.values()), f'Phiên {SESSION} chưa xong: {codes}'
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
        print("notebook khớp script" if same else "notebook lệch script: chạy python scripts/make_kaggle_notebook.py")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"đã ghi {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

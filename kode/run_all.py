"""Entry point tunggal UTS Machine Learning.

    python kode/run_all.py              # semua task (angka laporan + 3 CSV)
    python kode/run_all.py --task 2     # satu task saja (untuk reproduksi live)
    python kode/run_all.py --quick      # subsampel, hanya untuk debugging

Di akhir, ketiga CSV divalidasi oleh check_submission.py; jika ada yang
tidak sesuai format, proses keluar dengan kode error 1.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from importlib.metadata import version

import config
from common.utils import ensure_dirs, record_metric, set_seed

LIBS = ["numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "matplotlib"]


def print_versions() -> None:
    v = {"python": platform.python_version()}
    for lib in LIBS:
        v[lib] = version(lib)
    print("Versi:", ", ".join(f"{k} {x}" for k, x in v.items()))
    record_metric("lingkungan", "versi", v)
    record_metric("lingkungan", "seed", config.SEED)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["1", "2", "3", "all"], default="all")
    ap.add_argument("--quick", action="store_true", help="subsampel untuk debugging, bukan hasil final")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    set_seed()
    print_versions()
    t0 = time.perf_counter()

    if args.task in ("1", "all"):
        from task1_regresi import pipeline as t1
        t1.run(quick=args.quick)
    if args.task in ("2", "all"):
        from task2_fraud import pipeline as t2
        t2.run(quick=args.quick)
    if args.task in ("3", "all"):
        from task3_cluster import pipeline as t3
        t3.run(quick=args.quick)

    total = time.perf_counter() - t0
    print(f"Total waktu: {total / 60:.1f} menit")
    record_metric("waktu_detik", f"total_task_{args.task}", round(total, 1))

    if args.skip_check or args.quick or args.task != "all":
        return 0
    import check_submission
    return check_submission.main()


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
from pathlib import Path

from ecg_ai_monitor.data.simulator import save_simulated_csv
from ecg_ai_monitor.ml.train import train_model
from ecg_ai_monitor.screening.engine import analyze_ecg
from ecg_ai_monitor.utils.io import load_ecg_csv, save_json
from ecg_ai_monitor.utils.report import compact_report


def cmd_simulate(args: argparse.Namespace) -> None:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_simulated_csv(args.out, duration_s=args.duration, fs=args.fs, scenario=args.scenario, seed=args.seed)
    print(f"Saved simulated ECG CSV to {args.out}")


def cmd_analyze(args: argparse.Namespace) -> None:
    _, ecg, fs = load_ecg_csv(args.input, fs=args.fs)
    result = analyze_ecg(ecg, fs=fs, model_path=args.model).to_dict()
    if args.out:
        save_json(args.out, result)
        print(f"Saved analysis report to {args.out}")
    print(compact_report(result))


def cmd_train(args: argparse.Namespace) -> None:
    result = train_model(args.out, n_samples_per_class=args.n_samples, fs=args.fs, seed=args.seed)
    save_json(Path(args.out).with_suffix(".metrics.json"), result)
    print(f"Saved model to {args.out}")
    print(f"Saved metrics to {Path(args.out).with_suffix('.metrics.json')}")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("ecg_ai_monitor.api.main:app", host=args.host, port=args.port, reload=args.reload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wearable ECG AI monitoring toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="generate synthetic single-lead ECG CSV")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--fs", type=int, default=250)
    p.add_argument("--scenario", choices=["normal", "tachycardia", "bradycardia", "irregular", "mixed"], default="mixed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="data/sample_ecg.csv")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("analyze", help="analyze ECG CSV and generate screening report")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--fs", type=int, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--out", type=str, default="reports/report.json")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("train", help="train a lightweight ECG window classifier on synthetic data")
    p.add_argument("--out", type=str, default="models/ecg_window_classifier.joblib")
    p.add_argument("--n-samples", type=int, default=40, help="synthetic recordings per class")
    p.add_argument("--fs", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("serve", help="run FastAPI server")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

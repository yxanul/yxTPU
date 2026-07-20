"""Command-line entry point for standalone yxTPU pretraining."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from yxtpu_pretrain.config import ResolvedConfig, load_config


def _add_profiles(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="kda_hybrid_273m")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--data", default="synthetic")
    parser.add_argument("--hardware", default="v6e-8")
    parser.add_argument("--experiment", default="selected")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="DOTTED.KEY=VALUE",
        dest="overrides",
        help="override a resolved value; repeat for multiple values",
    )


def _resolve(args: argparse.Namespace) -> ResolvedConfig:
    return load_config(
        model=args.model,
        optimizer=args.optimizer,
        data=args.data,
        hardware=args.hardware,
        experiment=args.experiment,
        overrides=args.overrides,
    )


def _doctor(args: argparse.Namespace) -> int:
    from yxtpu_pretrain.runtime.doctor import run_doctor

    config = _resolve(args)
    ok, checks = run_doctor(config.hardware)
    for check in checks:
        mark = "OK" if check["ok"] else "FAIL"
        print(f"[{mark:4}] {check['name']}: {check['detail']}")
    print("Doctor does not provision, resize, or delete TPU resources.")
    return 0 if ok else 1


def _config_dump(args: argparse.Namespace) -> int:
    config = _resolve(args)
    if args.format == "json":
        print(json.dumps(config.as_dict(), indent=2))
    else:
        print(config.to_yaml(), end="")
    return 0


def _train(args: argparse.Namespace) -> int:
    from yxtpu_pretrain.train import run

    return run(_resolve(args))


def _benchmark(args: argparse.Namespace) -> int:
    from yxtpu_pretrain.train import run

    config = _resolve(args)
    if not config.experiment.benchmark:
        raise ValueError("benchmark subcommand requires a benchmark experiment profile")
    return run(config, benchmark_only=True)


def _profile(args: argparse.Namespace) -> int:
    from yxtpu_pretrain.train import run

    config = _resolve(args)
    if not config.experiment.profile_steps:
        config.experiment.profile_steps = (3, 4)
    return run(config, profile=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yx-pretrain")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="run standalone pretraining")
    _add_profiles(train)
    train.set_defaults(handler=_train)

    doctor = subparsers.add_parser("doctor", help="validate the local TPU/JAX environment")
    _add_profiles(doctor)
    doctor.set_defaults(handler=_doctor)

    config = subparsers.add_parser("config", help="configuration operations")
    config_subparsers = config.add_subparsers(dest="config_command", required=True)
    dump = config_subparsers.add_parser("dump", help="print the resolved configuration")
    _add_profiles(dump)
    dump.add_argument("--format", choices=("yaml", "json"), default="yaml")
    dump.set_defaults(handler=_config_dump)

    benchmark = subparsers.add_parser("benchmark", help="run a throughput benchmark profile")
    _add_profiles(benchmark)
    benchmark.set_defaults(handler=_benchmark)

    profile = subparsers.add_parser("profile", help="run with a JAX profiler capture")
    _add_profiles(profile)
    profile.set_defaults(handler=_profile)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    sys.exit(main())

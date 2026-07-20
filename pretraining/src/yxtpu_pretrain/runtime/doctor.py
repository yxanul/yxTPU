"""Environment validation without provisioning or mutating cloud resources."""

from __future__ import annotations

import importlib.metadata
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from yxtpu_pretrain.config import HardwareProfile


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _version(distribution: str) -> Check:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return Check(distribution, False, "not installed")
    return Check(distribution, True, version)


def _maxtext_pin() -> Check:
    package_root = Path(__file__).resolve().parents[3]
    expected = (package_root / "MAXTEXT_PIN").read_text(encoding="utf-8").strip()
    repository_root = package_root.parent
    try:
        imported_at = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "log",
                "-1",
                "--format=%H",
                f"--grep=import MaxText at {expected}",
                "--",
                "maxtext/pyproject.toml",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        clean = (
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository_root),
                    "diff",
                    "--quiet",
                    imported_at,
                    "--",
                    "maxtext",
                ],
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return Check("maxtext_pin", False, f"cannot inspect vendored MaxText: {error}")
    if not imported_at:
        return Check("maxtext_pin", False, f"no import commit records upstream {expected}")
    return Check(
        "maxtext_pin",
        clean,
        f"upstream {expected}, import commit {imported_at[:8]}, clean={clean}",
    )


def run_doctor(hardware: HardwareProfile | None = None) -> tuple[bool, list[dict[str, Any]]]:
    """Returns a machine-readable report and never creates TPU resources."""
    checks = [
        _version("jax"),
        _version("jaxlib"),
        _version("flax"),
        _version("optax"),
        _version("orbax-checkpoint"),
        _version("tokamax"),
        _version("maxtext"),
        _maxtext_pin(),
    ]
    try:
        import jax

        devices = jax.devices()
        platform = devices[0].platform if devices else "none"
        checks.append(Check("jax_devices", bool(devices), f"{len(devices)} device(s), {platform}"))
        if hardware is not None:
            checks.append(
                Check(
                    "hardware_device_count",
                    len(devices) == hardware.device_count,
                    (
                        f"profile {hardware.name} expects {hardware.device_count}, "
                        f"found {len(devices)}"
                    ),
                )
            )
            if hardware.performance_verified:
                checks.append(Check("performance_status", True, "v6e-8 certified profile"))
            else:
                checks.append(
                    Check("performance_status", True, "portable profile; not performance-certified")
                )
    except (
        Exception
    ) as error:  # TPU library initialization errors must be visible in doctor output.
        checks.append(Check("jax_devices", False, repr(error)))
    return all(check.ok for check in checks), [asdict(check) for check in checks]

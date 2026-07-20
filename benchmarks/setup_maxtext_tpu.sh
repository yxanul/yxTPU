#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/yxTPU}"
MAXTEXT_ROOT="${MAXTEXT_ROOT:-$WORKSPACE/maxtext}"
VENV="${VENV:-$WORKSPACE/.venv}"

if [[ ! -f "$MAXTEXT_ROOT/pyproject.toml" ]]; then
  echo "MaxText source tree not found at $MAXTEXT_ROOT." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || true)"
if [[ -z "$UV" ]]; then
  UV="$HOME/.local/bin/uv"
fi

"$UV" python install 3.12
"$UV" venv --python 3.12 --seed "$VENV"
source "$VENV/bin/activate"

cd "$MAXTEXT_ROOT"
"$UV" pip install --python "$VENV/bin/python" -e ".[tpu]" --resolution=lowest
install_tpu_pre_train_extra_deps

python - <<'PY'
import jax
import jaxlib
import flax
import optax

print(f"JAX:     {jax.__version__}")
print(f"jaxlib:  {jaxlib.__version__}")
print(f"Flax:    {flax.__version__}")
print(f"Optax:   {optax.__version__}")
print(f"Backend: {jax.default_backend()}")
print(f"Devices: {len(jax.devices())}")
print(f"Kinds:   {sorted({device.device_kind for device in jax.devices()})}")
PY


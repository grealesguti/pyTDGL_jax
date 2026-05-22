#!/usr/bin/env python3
"""
check_env.py — verify and optionally install all packages needed
for the py-tdgl → JAX migration.

Usage (from inside py-tdgl conda/venv):
    python check_env.py           # check only
    python check_env.py --install # install missing packages
"""

import importlib
import subprocess
import sys
import argparse

REQUIRED = [
    # (import_name, pip_name, min_version, notes)
    ("numpy",      "numpy>=1.24,<2.1",   "1.24.0", "JAX 0.4.x requires numpy<2.1"),
    ("scipy",      "scipy>=1.10",        "1.10.0", ""),
    ("jax",        "jax>=0.4.25",        "0.4.25", "CPU build; see below for GPU"),
    ("jaxlib",     "jaxlib>=0.4.25",     "0.4.25", "CPU build"),
    ("klujax",     "klujax>=0.2.0",      "0.2.0",  "sparse direct solver with AD (Phase 3)"),
    ("tdgl",       "tdgl>=0.8.0",        "0.8.0",  ""),
    ("h5py",       "h5py>=3.8",          "3.8.0",  ""),
    ("meshpy",     "meshpy",             "0",       ""),
    ("shapely",    "shapely>=2.0",       "2.0.0",  ""),
    ("matplotlib", "matplotlib>=3.7",    "3.7.0",  ""),
    ("numba",      "numba>=0.57",        "0.57.0", ""),
    ("tqdm",       "tqdm",               "0",       ""),
]

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def version_tuple(v):
    try:
        return tuple(int(x) for x in str(v).split(".")[:3])
    except Exception:
        return (0,)

def check_package(import_name, min_version):
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "unknown")
        if min_version == "0" or version_tuple(ver) >= version_tuple(min_version):
            return "ok", ver
        else:
            return "old", ver
    except ImportError:
        return "missing", None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true",
                        help="pip-install any missing or outdated packages")
    args = parser.parse_args()

    print(f"\n{BOLD}Python:{RESET} {sys.executable}  ({sys.version.split()[0]})\n")
    print(f"{'Package':<14} {'Status':<10} {'Version':<12} {'Notes'}")
    print("─" * 70)

    to_install = []
    all_ok = True

    for import_name, pip_spec, min_ver, notes in REQUIRED:
        status, ver = check_package(import_name, min_ver)
        ver_str = ver or "—"
        if status == "ok":
            color = GREEN
            label = "✓ ok"
        elif status == "old":
            color = YELLOW
            label = "⚠ old"
            to_install.append(pip_spec)
            all_ok = False
        else:
            color = RED
            label = "✗ missing"
            to_install.append(pip_spec)
            all_ok = False
        print(f"  {import_name:<12} {color}{label:<10}{RESET} {ver_str:<12} {notes}")

    print()

    # Special: warn if numpy>=2.0 (breaks JAX 0.4.x)
    try:
        import numpy as np
        if version_tuple(np.__version__) >= (2, 1, 0):
            print(f"{YELLOW}⚠  numpy {np.__version__} detected — JAX 0.4.x requires numpy<2.1{RESET}")
            print(f"   Fix: pip install 'numpy>=1.24,<2.1'")
            all_ok = False
    except ImportError:
        pass

    if all_ok:
        print(f"{GREEN}{BOLD}All packages OK.{RESET}")
    else:
        print(f"{YELLOW}Missing/outdated packages:{RESET} {', '.join(to_install)}")

    if to_install and args.install:
        print(f"\n{BOLD}Installing...{RESET}")
        cmd = [sys.executable, "-m", "pip", "install"] + to_install
        print("  " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"\n{GREEN}Done. Re-run without --install to verify.{RESET}")
    elif to_install and not args.install:
        print(f"\nRun with {BOLD}--install{RESET} to install automatically, or:")
        print(f"  pip install -r requirements-jax.txt")

    # GPU hint
    print(f"""
{BOLD}GPU note:{RESET}
  WSL2 + CUDA 12 →  pip install --upgrade "jax[cuda12]"
  WSL2 + CUDA 11 →  pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  CPU only       →  current install is fine

{BOLD}Quick sanity check after installing:{RESET}
  python -c "import jax; import jax.numpy as jnp; x = jnp.ones(4); print(jax.devices()); print(x)"
""")

if __name__ == "__main__":
    main()

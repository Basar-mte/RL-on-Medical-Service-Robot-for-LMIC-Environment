"""Build Isaac_Package/ -- a portable folder to copy onto the Isaac Sim machine.

Mirrors the repository layout deliberately. Every script in deploy/isaac/
resolves the project root by walking up a fixed number of directories, so
keeping the same relative depth means not a single path constant has to be
edited on the target machine.

    python scripts/build_isaac_package.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "Isaac_Package"
DEMO_MODELS = ROOT / "deploy" / "gazebo_demo_package" / "models"

# Source packages copied whole: all small, and copying them entire removes any
# risk of a missed transitive import on the target machine.
CODE_DIRS = ["envs", "eval", "rewards", "baselines"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src: Path, dst: Path, skip_cache=True) -> int:
    n = 0
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        if skip_cache and ("__pycache__" in p.parts or p.suffix == ".pyc"):
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
    return n


def main() -> None:
    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir(parents=True)

    manifest = {
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Isaac Sim evaluation testbed for LMIC-HospitalNav Tier 2",
        "note": "Evaluation only -- no training, no benchmark result depends on this.",
        "contents": {},
    }

    # ---- code (same relative layout as the repo) ------------------------
    for d in CODE_DIRS:
        n = copy_tree(ROOT / d, PKG / d)
        manifest["contents"][d] = f"{n} files"
        print(f"  {d:<22} {n} files")

    n = copy_tree(ROOT / "deploy" / "isaac", PKG / "deploy" / "isaac")
    manifest["contents"]["deploy/isaac"] = f"{n} files"
    print(f"  {'deploy/isaac':<22} {n} files (scripts, URDF, meshes, USD stage)")

    # ---- config: the single source the world and env both read ----------
    (PKG / "configs").mkdir(exist_ok=True)
    cfg = ROOT / "configs" / "hospital_v4.yaml"
    shutil.copy2(cfg, PKG / "configs" / cfg.name)
    manifest["contents"]["configs/hospital_v4.yaml"] = sha256(cfg)
    print(f"  {'configs':<22} hospital_v4.yaml  sha256 {sha256(cfg)[:12]}...")

    # ---- models: reuse the vetted pairs from the demo package -----------
    (PKG / "models").mkdir(exist_ok=True)
    models = {}
    if DEMO_MODELS.exists():
        for f in sorted(DEMO_MODELS.iterdir()):
            if f.suffix in (".zip", ".pkl"):
                shutil.copy2(f, PKG / "models" / f.name)
                models[f.name] = sha256(f)
        print(f"  {'models':<22} {len(models)} files "
              f"({len([k for k in models if k.endswith('.zip')])} policies + matched vecnorms)")
    else:
        print("  !! gazebo_demo_package/models not found -- copy checkpoints manually")
    manifest["contents"]["models"] = models

    (PKG / "requirements.txt").write_text(
        "# Install into the Isaac Sim Python environment, not a fresh venv.\n"
        "stable-baselines3>=2.0\n"
        "gymnasium>=0.29\n"
        "numpy\n"
        "pyyaml\n"
        "torch  # normally already present in the Isaac Python env\n",
        encoding="utf-8")

    # Package docs are versioned in deploy/isaac/package_docs/ -- Isaac_Package/
    # is gitignored and wiped on every rebuild, so anything authored there alone
    # would be both untracked and lost.
    docs = ROOT / "deploy" / "isaac" / "package_docs"
    n_docs = 0
    for md in sorted(docs.glob("*.md")):
        shutil.copy2(md, PKG / md.name)
        n_docs += 1
        print(f"  {'doc':<22} {md.name}")
    manifest["contents"]["docs"] = f"{n_docs} files from deploy/isaac/package_docs/"

    (PKG / "MANIFEST.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")

    total = sum(f.stat().st_size for f in PKG.rglob("*") if f.is_file())
    print(f"\nIsaac_Package/ built: "
          f"{sum(1 for f in PKG.rglob('*') if f.is_file())} files, {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()

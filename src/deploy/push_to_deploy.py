"""Utility script to sync the local repo to the deploy branch and push."""
from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path

DEFAULT_MESSAGE = "chore: sync deploy"


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)


@lru_cache(maxsize=1)
def load_branch(repo: Path) -> str:
    config_path = repo / "src" / "user" / "config.yaml"
    if not config_path.exists():
        return "deploy"

    try:
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise RuntimeError(
            "omegaconf is required to read src/user/config.yaml. Install hydra-core/omegaconf."
        ) from exc

    cfg = OmegaConf.load(config_path)
    branch = cfg.get("repo", {}).get("deploy_branch", "deploy")
    return branch


def ensure_branch(repo: Path, branch: str) -> None:
    try:
        run(["git", "rev-parse", "--verify", branch], repo)
    except subprocess.CalledProcessError:
        run(["git", "switch", "-c", branch], repo)
    else:
        run(["git", "switch", branch], repo)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    deploy_branch = load_branch(repo)

    try:
        run(["git", "fetch", "origin"], repo)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr)
        return exc.returncode

    try:
        ensure_branch(repo, deploy_branch)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr)
        return exc.returncode

    try:
        run(["git", "merge", "--ff-only", "origin/" + deploy_branch], repo)
    except subprocess.CalledProcessError:
        # Branch might not exist remotely yet; ignore fast-forward failures.
        pass

    run(["git", "add", "-A"], repo)

    status = run(["git", "status", "--porcelain"], repo)
    if not status.stdout.strip():
        print("No changes to commit; deploy branch is up to date.")
        return 0

    try:
        run(["git", "commit", "-m", DEFAULT_MESSAGE], repo)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr)
        return exc.returncode

    try:
        push = run(["git", "push", "origin", deploy_branch], repo)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr)
        return exc.returncode

    print(push.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

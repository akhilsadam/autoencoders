"""Deploy helper that pushes the deploy branch and schedules a Slurm job via SSH."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .push_to_deploy import main as push_to_deploy

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "src" / "user" / "config.yaml"


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def read_config(path: Path) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"User config not found at {path}")
    return OmegaConf.load(path)


def build_remote_script(cfg: DictConfig) -> str:
    ssh_cfg = cfg.ssh
    repo_cfg = cfg.repo
    path_cfg = cfg.paths
    train_cfg = cfg.train
    slurm_cfg = cfg.slurm

    remote_workdir = Path(path_cfg.remote_workdir).as_posix()
    remote_repo_url = repo_cfg.url
    deploy_branch = repo_cfg.deploy_branch

    entrypoint = train_cfg.entrypoint
    overrides = train_cfg.get("overrides")
    if overrides:
        entrypoint = f"{entrypoint} {overrides}"

    exports = {
        "REPO_URL": remote_repo_url,
        "DEPLOY_BRANCH": deploy_branch,
        "REMOTE_WORKDIR": remote_workdir,
    "OUTPUT_DIR": Path(path_cfg.remote_output_dir).as_posix(),
    "LOCAL_SYNC_TARGET": Path(path_cfg.local_sync_dir).as_posix(),
    "DATA_DIR": Path(path_cfg.remote_data_dir).as_posix(),
        "ENTRYPOINT": entrypoint,
        "WANDB_MODE": "online",
    }

    export_lines = [f"export {key}={shlex.quote(str(value))}" for key, value in exports.items()]

    sbatch_parts = ["sbatch"]
    if slurm_cfg.get("partition"):
        sbatch_parts += ["--partition", shlex.quote(str(slurm_cfg.partition))]
    if slurm_cfg.get("account"):
        sbatch_parts += ["--account", shlex.quote(str(slurm_cfg.account))]
    if slurm_cfg.get("qos"):
        sbatch_parts += ["--qos", shlex.quote(str(slurm_cfg.qos))]
    if slurm_cfg.get("gres"):
        sbatch_parts += ["--gres", shlex.quote(str(slurm_cfg.gres))]
    if slurm_cfg.get("time"):
        sbatch_parts += ["--time", shlex.quote(str(slurm_cfg.time))]
    if slurm_cfg.get("job_name"):
        sbatch_parts += ["--job-name", shlex.quote(str(slurm_cfg.job_name))]
    for flag in slurm_cfg.get("additional_flags", []) or []:
        sbatch_parts.append(str(flag))
    sbatch_parts.append("src/deploy/slurm_template.sh")
    sbatch_cmd = " ".join(sbatch_parts)

    remote_data_dir = Path(path_cfg.remote_data_dir).as_posix()

    remote_lines = [
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(remote_workdir)}",
        f"mkdir -p {shlex.quote(remote_data_dir)}",
        f"if [ ! -d {shlex.quote(remote_workdir)}/.git ]; then",
        f"  git clone {shlex.quote(remote_repo_url)} {shlex.quote(remote_workdir)}",
        "fi",
        f"cd {shlex.quote(remote_workdir)}",
        f"git fetch origin {shlex.quote(deploy_branch)}",
        f"git checkout {shlex.quote(deploy_branch)}",
        f"git pull --ff-only origin {shlex.quote(deploy_branch)}",
    ]
    remote_lines.extend(export_lines)
    remote_lines.append(sbatch_cmd)
    return "\n".join(remote_lines)


def build_ssh_command(cfg: DictConfig, remote_script: str) -> list[str]:
    ssh_cfg = cfg.ssh
    user_host = f"{ssh_cfg.user}@{ssh_cfg.host}"

    ssh_cmd = ["ssh"]
    if ssh_cfg.get("identity_file"):
        ssh_cmd += ["-i", str(Path(ssh_cfg.identity_file).expanduser())]
    if ssh_cfg.get("port") and ssh_cfg.port != 22:
        ssh_cmd += ["-p", str(ssh_cfg.port)]

    ssh_cmd.append(user_host)
    ssh_cmd.append("bash -lc " + shlex.quote(remote_script))
    return ssh_cmd


def deploy(cfg_path: Path, dry_run: bool) -> int:
    cfg = read_config(cfg_path)

    # Push local changes before launching remote jobs.
    push_rc = push_to_deploy()
    if push_rc != 0:
        return push_rc

    remote_script = build_remote_script(cfg)
    ssh_cmd = build_ssh_command(cfg, remote_script)

    if dry_run:
        print("SSH command:", " ".join(ssh_cmd))
        print("Remote script:\n" + remote_script)
        return 0

    try:
        result = run(ssh_cmd)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr)
        return exc.returncode

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy training job to Slurm cluster over SSH")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to user config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return deploy(args.config, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

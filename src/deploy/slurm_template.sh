#!/bin/bash
#SBATCH --job-name=ae-train
#SBATCH --partition=PARTITION_NAME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Stop on first failure and fail pipelines when any step fails.
set -euo pipefail

# User-configurable variables.
REPO_URL=${REPO_URL:-git@github.com:your-org/autoencoders.git}
DEPLOY_BRANCH=${DEPLOY_BRANCH:-deploy}
REMOTE_WORKDIR=${REMOTE_WORKDIR:-$SCRATCH/autoencoders}
ENTRYPOINT=${ENTRYPOINT:-python -m src.autoencoders.train}
OUTPUT_DIR=${OUTPUT_DIR:-$REMOTE_WORKDIR/runs}
LOCAL_SYNC_TARGET=${LOCAL_SYNC_TARGET:-/path/to/local/autoencoders}
DATA_DIR=${DATA_DIR:-$REMOTE_WORKDIR/data}

mkdir -p "$DATA_DIR"
export DATA_DIR

# Optionally load modules or activate an environment here.
# module load cuda/12.1
# source ~/.virtualenvs/autoencoders/bin/activate

if [ ! -d "$REMOTE_WORKDIR" ]; then
    echo "Creating remote workdir at $REMOTE_WORKDIR"
    mkdir -p "$REMOTE_WORKDIR"
    git clone "$REPO_URL" "$REMOTE_WORKDIR"
fi

cd "$REMOTE_WORKDIR"

echo "Pulling latest changes from $DEPLOY_BRANCH"
git fetch origin "$DEPLOY_BRANCH"
git checkout "$DEPLOY_BRANCH"
git pull --ff-only origin "$DEPLOY_BRANCH"

# Ensure dependencies are available. This is a no-op if already installed.
if [ -f src/install/requirements.txt ]; then
    pip install --user -r src/install/requirements.txt
fi

# Export WANDB API key if needed.
# export WANDB_API_KEY=your_api_key
# Propagate Hydra data root so local + remote agree.

RUN_ID=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="$OUTPUT_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-slurm}
export WANDB_NAME="slurm-${RUN_ID}"
export WANDB_PROJECT=${WANDB_PROJECT:-autoencoder-deploy}
export HYDRA_FULL_ERROR=1
export WANDB_MODE=${WANDB_MODE:-online}

echo "Starting training run"
HYDRA_OVERRIDES="paths.data_root=$DATA_DIR paths.artifacts_root=$RUN_DIR/artifacts wandb.project=$WANDB_PROJECT wandb.mode=$WANDB_MODE wandb.group=$WANDB_RUN_GROUP run.name=$WANDB_NAME run.tags=[slurm,deploy] hydra.run.dir=$RUN_DIR/hydra"
set -x
eval "$ENTRYPOINT $HYDRA_OVERRIDES" | tee "$RUN_DIR/train.log"
set +x

# Collect artifacts (images + text) back to local target.
echo "Syncing outputs back to $LOCAL_SYNC_TARGET"
rsync -av --relative \
    --include="*/" \
    --include="*.png" \
    --include="*.jpg" \
    --include="*.jpeg" \
    --include="*.txt" \
    --include="*.log" \
    --include="*.yaml" \
    --include="*.ckpt" \
    --exclude="*" \
    "$RUN_DIR" "$LOCAL_SYNC_TARGET"

echo "Job finished"

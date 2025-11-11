from git import Repo
from omegaconf import DictConfig, OmegaConf


def _compute_diff(cfg: DictConfig) -> None:
    """Calculate and save git changes to output directory."""
    repo = Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    dirty = repo.is_dirty()

    if dirty:
        diff = repo.git.diff()
    else:
        # diff to last commit
        diff = repo.git.diff(f"{sha}~1", sha)

    try:
        from .util import llm
        summary, short_sum = llm.summarize_diff(diff)
        print(f"\nLong changelog message:{summary}\n")
        print(f"\nShort changelog message:{short_sum}\n")
    except Exception as e:
        print(f"Warning: failed to summarize diff: {e}")
        summary = ""
        short_sum = ""
        
    return {
        'sha': sha,
        'dirty': dirty,
        'long_msg': summary,
        'short_msg': short_sum,
    }
    
OmegaConf.register_new_resolver("gitinfo", _compute_diff)
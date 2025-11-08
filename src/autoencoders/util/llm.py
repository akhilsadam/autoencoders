"""
Lightweight local LLM utilities for summarizing git diffs.

Uses GPT4All with a small instruction-tuned model (e.g. Mistral-7B-Instruct).
No internet connection is required after the model is downloaded once.
"""

import os
from typing import Optional

try:
    from gpt4all import GPT4All
except ImportError as e:
    raise ImportError(
        "GPT4All is required for local LLM summarization. "
        "Install with: pip install gpt4all"
    ) from e


# Default local model — small and CPU-friendly.
# You can change this to any .gguf model that GPT4All supports.
_DEFAULT_MODEL = os.environ.get(
    "LLM_MODEL",
    "mistral-7b-instruct-v0.1.Q4_0.gguf",
)


def summarize_diff(diff_text: str, max_chars: int = 4000) -> str:
    """
    Summarize a git diff into a concise changelog entry using a local LLM.

    Args:
        diff_text: Raw output from `git diff`.
        max_chars: Optional limit on diff length to prevent huge prompts.

    Returns:
        A short human-readable changelog string.
    """
    if not diff_text.strip():
        return "No code changes detected."

    # Truncate to avoid feeding massive diffs
    prompt = (
        "Summarize the following git diff into a single concise changelog entry. "
        "Use imperative tone (e.g., 'Fix', 'Add', 'Refactor', 'Improve'). "
        "If multiple changes exist, merge them into one short coherent summary.\n\n"
        f"{diff_text[:max_chars]}"
    )

    model = GPT4All(_DEFAULT_MODEL)
    response: Optional[str] = None
    try:
        response = model.generate(prompt, max_tokens=100, temp=0.2).strip()
    except Exception as e:
        response = f"(LLM summarization failed: {e})"

    return response or "No summary generated."

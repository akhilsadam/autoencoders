# llm.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _extract_diff_content(diff_text: str) -> str:
    """Extract meaningful lines from a git diff for summarization."""
    content_lines = []
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            content_lines.append(line[1:].strip())
        elif line.startswith("-") and not line.startswith("---"):
            content_lines.append(f"Removed: {line[1:].strip()}")
    return "\n".join(content_lines)


def _fallback_summarizer(diff_text: str, max_chars: int = 4000) -> str:
    if not diff_text.strip():
        return "No code changes detected."

    try:
        from transformers import pipeline
    except ImportError:
        return "Diff detected, but transformers not installed to summarize it."

    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1,  # CPU
    )

    # Extract only meaningful diff lines
    content = _extract_diff_content(diff_text)

    # Split into chunks if too large
    def chunk_text(text, size=max_chars):
        lines, chunk, chunks = text.splitlines(), [], []
        total = 0
        for line in lines:
            total += len(line)
            chunk.append(line)
            if total >= size:
                chunks.append("\n".join(chunk))
                chunk = []
                total = 0
        if chunk:
            chunks.append("\n".join(chunk))
        return chunks

    chunks = chunk_text(content)
    summaries = []
    for c in chunks:
        prompt = (
            "Summarize the following code changes in plain English as a short changelog entry. "
            "Focus on functional changes and ignore whitespace or formatting changes:\n\n"
            f"{c}"
        )
        out = summarizer(prompt, max_length=60, min_length=10, do_sample=False)
        summaries.append(out[0]["summary_text"].strip())

    return " ".join(summaries)


def summarize_diff(diff_text: str) -> str:
    """Try GPT4All first, fallback to transformers summarizer if unavailable."""
    if not diff_text.strip():
        return "No code changes detected."

    # Try GPT4All if installed and compatible
    try:
        from gpt4all import GPT4All

        # Load a lightweight model (adjust path to your local model if needed)
        model_path = os.environ.get("GPT4ALL_MODEL_PATH", "ggml-model-gpt4all-j-v1.3-groovy.bin")
        llm = GPT4All(model_path)

        content = _extract_diff_content(diff_text)
        prompt = (
            "Summarize the following code changes in plain English as a short changelog entry. "
            "Focus on functional changes and ignore whitespace or formatting changes:\n\n"
            f"{content[:4000]}"
        )
        summary = llm.generate(prompt, streaming=False)
        if summary.strip():
            return summary.strip()
    except Exception as e:
        print(f"Info: GPT4All unavailable, falling back to Transformers: {e}")

    return _fallback_summarizer(diff_text)

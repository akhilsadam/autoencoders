# llm.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

    # Split into chunks
    def chunk_diff(text, size=max_chars):
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

    chunks = chunk_diff(diff_text)
    summaries = [summarizer(c, max_length=60, min_length=10, do_sample=False)[0]["summary_text"].strip()
                 for c in chunks]
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
        prompt = f"Summarize this git diff as a short changelog:\n\n{diff_text[:4000]}"
        summary = llm.generate(prompt, streaming=False)
        if summary.strip():
            return summary.strip()
    except Exception as e:
        # GPT4All failed (likely GLIBC or missing binary)
        print(f"Info: GPT4All unavailable, falling back to Transformers: {e}")

    # Fallback
    return _fallback_summarizer(diff_text)

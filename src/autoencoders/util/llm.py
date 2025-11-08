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


def _chunk_text(text: str, max_chars: int = 4000) -> list[str]:
    """Split text into chunks of roughly max_chars for the model."""
    lines, chunk, chunks = text.splitlines(), [], []
    total = 0
    for line in lines:
        total += len(line)
        chunk.append(line)
        if total >= max_chars:
            chunks.append("\n".join(chunk))
            chunk, total = [], 0
    if chunk:
        chunks.append("\n".join(chunk))
    return chunks


def _fallback_summarizer(diff_text: str, max_chars: int = 4000) -> str:
    """Use a CPU-friendly Transformers model as fallback."""
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

    content = _extract_diff_content(diff_text)
    chunks = _chunk_text(content, size=max_chars)
    summaries = []

    for c in chunks:
        prompt = (
            "Summarize the following code changes in plain English as a short changelog entry. "
            "Focus on functional changes and ignore whitespace or formatting changes:\n\n"
            f"{c}"
        )
        out = summarizer(prompt, max_length=80, min_length=15, do_sample=False)
        summaries.append(out[0]["summary_text"].strip())

    return " ".join(summaries)


def summarize_diff(diff_text: str) -> str:
    """Summarize a git diff using Falcon-7B-Instruct or fallback to CPU summarizer."""
    if not diff_text.strip():
        return "No code changes detected."

    content = _extract_diff_content(diff_text)
    chunks = _chunk_text(content)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        model_name = "tiiuae/falcon-7b-instruct"
        device = 0 if torch.cuda.is_available() else -1

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device == 0 else None,
            torch_dtype=torch.float16 if device == 0 else torch.float32,
        )

        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
        summaries = []

        for c in chunks:
            prompt = (
                "Summarize the following code changes in plain English as a short changelog entry. "
                "Focus on functional changes and ignore whitespace or formatting changes:\n\n"
                f"{c}"
            )
            out = generator(prompt, max_new_tokens=150, do_sample=False)
            text = out[0]["generated_text"].strip()
            # Remove the prompt from the output if present
            text = text.replace(prompt, "").strip()
            summaries.append(text)

        return " ".join(summaries)

    except Exception as e:
        print(f"Info: Falcon-7B-Instruct unavailable, falling back to CPU summarizer: {e}")
        return _fallback_summarizer(diff_text)

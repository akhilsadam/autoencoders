# llm.py 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def _extract_diff_content(diff_text: str) -> str:
    """Extract meaningful lines from a git diff for summarization."""
    content_lines = []
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            content_lines.append(line.strip())
        elif line.startswith("-") and not line.startswith("---"):
            content_lines.append(line.strip())
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
            chunk = []
            total = 0
    if chunk:
        chunks.append("\n".join(chunk))
    return chunks 

def summarize_diff(diff_text: str, quality=1) -> tuple[str, str]:
    """Summarize a git diff into a long changelog and a short label."""
    if not diff_text.strip():
        return "No code changes detected.", ""

    content = _extract_diff_content(diff_text)
    chunks = _chunk_text(content)

    print(f"Summarizing diff in {len(chunks)} chunks...")
    print(f"Diff content preview:\n{content[:500]}...\n")
    print(f"First chunk preview:\n{chunks[0][:500]}...\n") 
    print(f"Last chunk preview:\n{chunks[-1][:500]}...\n")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        model_name = "tiiuae/Falcon-H1-1.5B-Instruct" if quality == 1 else "tiiuae/falcon-7b-instruct"
        device = 0 if torch.cuda.is_available() else -1

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

        # Step 1: Long summary
        summaries = []
        for c in chunks:
            prompt = (
                "Summarize the following code changes in plain English as a concise but descriptive changelog. "
                "+ denotes additions, - denotes deletions."
                "Focus on functional changes, ignore whitespace/formatting changes. Do NOT repeat code or words.\n\n"
                f"{c}"
            )
            out = generator(prompt, max_new_tokens=150, do_sample=False)
            text = out[0]["generated_text"].strip().replace(prompt, "").strip()
            summaries.append(text)

        long_summary = " ".join(summaries).replace("\n", " ").strip()

        # Step 2: Short label
        label_prompt = (
            "Generate a 3-4 word concise label summarizing the following changelog. "
            "Focus on functional changes only. Make it folder-safe (no punctuation, use underscores or CamelCase):\n\n"
            f"{long_summary}\n\n"
            "Output only the label:"
        )
        label_out = generator(label_prompt, max_new_tokens=20, do_sample=False)
        short_summary = label_out[0]["generated_text"].strip().replace(label_prompt, "").strip()

        # Clean up label
        short_summary = (
            short_summary.replace(" ", "_")
            .replace("-", "_")
            .replace(".", "")
            .upper()
        )
        if not short_summary:
            # fallback
            short_summary = "_".join(long_summary.split()[:3]).upper()

        return long_summary, short_summary

    except Exception as e:
        print(f"Info: LLM unavailable, skipping: {e}")
        # return _fallback_summarizer(diff_text), "FALLBACK"
        return "", ""

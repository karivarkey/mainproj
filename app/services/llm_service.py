from pathlib import Path
from llama_cpp import Llama
from app.config import LLM_DIR
from app.services.cache_service import model_cache, save_cache

current_llm: Llama | None = None
current_name: str | None = None
current_path: Path | None = None


def local_gguf_path(model_name: str) -> Path:
    safe = model_name.replace("/", "__")
    return LLM_DIR / f"{safe}.gguf"


def download_gguf(url: str, model_name: str) -> Path:
    import requests, os

    LLM_DIR.mkdir(parents=True, exist_ok=True)
    dest = local_gguf_path(model_name)

    if dest.exists():
        if model_name not in model_cache["llms"]:
            model_cache["llms"].append(model_name)
            save_cache(model_cache)
        return dest

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0"))
        downloaded = 0

        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=2*1024*1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"{pct}% {downloaded/1e6:.1f}/{total/1e6:.1f} MB", end="\r")

    model_cache["llms"].append(model_name)
    save_cache(model_cache)
    return dest


def load_llm(model_name: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
    global current_llm, current_name, current_path

    gguf_path = local_gguf_path(model_name)
    if not gguf_path.exists():
        raise FileNotFoundError(gguf_path)

    print(f"Loading GGUF: {gguf_path}")

    # Kill previous model (free GPU VRAM)
    current_llm = None

    current_llm = Llama(
        model_path=str(gguf_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,  # -1 = max GPU offload
        verbose=False,
    )

    current_name = model_name
    current_path = gguf_path

    print("Model loaded.")
    return current_llm

# Alias for compatibility with routes.py
load_llm_from_gguf = load_llm

# Hardcoded server URL constant for compatibility
SERVER_URL = "http://localhost:8080"


def unload_llm():
    global current_llm, current_name, current_path
    current_llm = None
    current_name = None
    current_path = None
    print("LLM unloaded.")


def reset_llm_context():
    """Clear the KV cache and context from the current LLM to free memory."""
    global current_llm
    if current_llm is None:
        return
    try:
        current_llm.reset()
        print("[LLM] Context cleared.")
    except Exception as e:
        print(f"[LLM] Warning: Could not reset context: {e}")


def llm_generate(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    stop: list[str] | None = None,
) -> str:
    if current_llm is None:
        raise RuntimeError("LLM not loaded")

    stop = stop or ["</s>", "```"]

    try:
        response = current_llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop,
            echo=False,
        )
        result = response["choices"][0]["text"]
        # Clear context after generation to prevent buildup
        reset_llm_context()
        return result
    except RuntimeError as e:
        if "llama_decode returned -1" in str(e):
            raise RuntimeError(
                f"LLM generation failed: context or memory limit exceeded.\n"
                f"Try reducing max_new_tokens (currently {max_new_tokens}) or "
                f"increasing n_ctx when loading the model."
            ) from e
        raise


def llm_generate_stream(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    stop: list[str] | None = None,
):
    """
    Stream generation from the current Llama instance.
    Yields text chunks (strings) as they are produced by the model.
    This is a best-effort wrapper that handles different stream chunk shapes
    returned by `llama_cpp` across versions.
    """
    if current_llm is None:
        raise RuntimeError("LLM not loaded")

    # The llama_cpp client supports streaming; calling with stream=True
    # returns an iterator of chunks (often dicts). We yield the text
    # content from each chunk.
    stop = stop or ["</s>", "```"]

    try:
        for chunk in current_llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop,
            echo=False,
            stream=True,
        ):
            try:
                # chunk may be a dict with choices -> text, or a simple string
                if isinstance(chunk, dict):
                    choices = chunk.get("choices") or []
                    if choices:
                        # Common shape: {'choices': [{'text': '...'}]}
                        text = choices[0].get("text") or ""
                    else:
                        # Fallback: try top-level text
                        text = chunk.get("text", "")
                elif isinstance(chunk, str):
                    text = chunk
                else:
                    text = str(chunk)
            except Exception:
                # Ignore malformed chunks
                continue

            if text:
                yield text
        
        # Clear context after streaming generation completes
        reset_llm_context()
    except RuntimeError as e:
        if "llama_decode returned -1" in str(e):
            raise RuntimeError(
                f"LLM streaming generation failed: context or memory limit exceeded.\n"
                f"Try reducing max_new_tokens (currently {max_new_tokens}) or "
                f"increasing n_ctx when loading the model."
            ) from e
        raise

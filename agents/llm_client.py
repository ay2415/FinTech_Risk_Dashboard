import os
import json
import urllib.request
import urllib.error

_OLLAMA_MODELS = None

def get_installed_ollama_models(timeout: float = 0.2) -> list:
    global _OLLAMA_MODELS
    if _OLLAMA_MODELS is not None:
        return _OLLAMA_MODELS
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            _OLLAMA_MODELS = [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        _OLLAMA_MODELS = []
    return _OLLAMA_MODELS

def check_ollama_available(url: str = "http://127.0.0.1:11434", timeout: float = 0.2) -> bool:
    return len(get_installed_ollama_models(timeout=timeout)) > 0

def query_llm(prompt: str, system_prompt: str = "", model: str = "llama3.1:8b", timeout: float = 2.0) -> str:
    """
    Lightweight, dependency-free LLM caller.
    Prioritizes local Ollama, then OpenAI API if key is present, 
    with a graceful deterministic fallback if no service is running.
    """
    # 1. Try Local Ollama (http://127.0.0.1:11434)
    installed_models = get_installed_ollama_models()
    model_matched = any(m == model or m.startswith(model.split(":")[0]) for m in installed_models)
    
    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
    if model_matched:
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2
                }
            }
            req = urllib.request.Request(
                ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    result = json.loads(response.read().decode("utf-8"))
                    return result.get("response", "").strip()
        except Exception:
            pass

    # 2. Try OpenAI API if OPENAI_API_KEY is available
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            openai_url = "https://api.openai.com/v1/chat/completions"
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.2
            }
            req = urllib.request.Request(
                openai_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_key}"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    result = json.loads(response.read().decode("utf-8"))
                    return result["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    # 3. Graceful Fallback if LLM is offline
    return ""

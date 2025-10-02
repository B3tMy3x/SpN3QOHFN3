from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
import httpx  # type: ignore


def _find_model_dir(base: str) -> str:
    """
    Attempt to find a HF model directory under `base` that has a config.json.
    Returns `base` if it itself is a model dir.
    """
    base = os.path.expanduser(base)
    if os.path.isfile(os.path.join(base, "config.json")):
        return base
    # Try immediate children
    if os.path.isdir(base):
        for name in os.listdir(base):
            cand = os.path.join(base, name)
            if os.path.isfile(os.path.join(cand, "config.json")):
                return cand
    raise FileNotFoundError(f"No model config.json found under: {base}")


@dataclass
class GenerationConfig:
    temperature: float = 0.15
    top_p: float = 0.9
    max_new_tokens: int = 512
    do_sample: bool = False


class LLM:
    """
    Local Transformers-based chat wrapper for DeepSeek Coder 6.7B (instruct).

    - Does not use any external API.
    - Loads model from local path (default: kaggle/input/deepseek-coder or /kaggle/input/deepseek-coder).
    - Uses tokenizer chat template if available; otherwise falls back to a simple role-tagged format.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        gen: Optional[GenerationConfig] = None,
        dtype: Optional[str] = None,  # "auto"|"float16"|"bfloat16"|"float32"
    ) -> None:
        # Load generation parameters (allow env overrides)
        if gen is not None:
            self.gen = gen
        else:
            try:
                t = float(os.getenv("LLM_TEMPERATURE", "0.15"))
            except Exception:
                t = 0.15
            try:
                tp = float(os.getenv("LLM_TOP_P", "0.9"))
            except Exception:
                tp = 0.9
            try:
                mxt = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))
            except Exception:
                mxt = 512
            self.gen = GenerationConfig(temperature=t, top_p=tp, max_new_tokens=mxt, do_sample=False)
        self.remote_openrouter = str(os.getenv("IS_OPENROUTER_MODEL", "false")).lower() in ("1", "true", "yes", "on")
        self.remote_ollama = str(os.getenv("IS_OLLAMA_MODEL", "false")).lower() in ("1", "true", "yes", "on")
        # Explicit backend override: local | ollama | openrouter
        backend = os.getenv("LLM_BACKEND")
        if backend:
            b = backend.strip().lower()
            if b == "local":
                self.remote_openrouter = False
                self.remote_ollama = False
            elif b == "ollama":
                self.remote_openrouter = False
                self.remote_ollama = True
            elif b == "openrouter":
                self.remote_openrouter = True
                self.remote_ollama = False
        # If both are accidentally true, prefer Ollama to avoid unexpected OpenRouter use
        if self.remote_openrouter and self.remote_ollama:
            self.remote_openrouter = False
        self._tok = None
        self._model = None
        self._openrouter_model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-coder:6.7b-instruct")
        self._openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self._ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._ollama_model = os.getenv("OLLAMA_MODEL", "gpt-oss:latest")
        # Reused HTTP client for remote calls to avoid repeated handshakes
        self._http_client: Optional[httpx.Client] = None

        if not (self.remote_openrouter or self.remote_ollama):
            # Resolve model directory
            local_hint = os.getenv("LOCAL_MODEL_DIR")
            default_candidates = [
                model_dir,
                local_hint,
                "kaggle/input/deepseek-coder",
                "/kaggle/input/deepseek-coder",
            ]
            chosen = None
            for cand in default_candidates:
                if not cand:
                    continue
                try:
                    chosen = _find_model_dir(cand)
                    break
                except Exception:
                    continue
            if not chosen:
                raise FileNotFoundError(
                    "Local model not found. Set LOCAL_MODEL_DIR or pass model_dir, or set IS_OPENROUTER_MODEL=true."
                )

            # dtype/device strategy
            if dtype is None:
                if torch.cuda.is_available():
                    self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                else:
                    self.torch_dtype = torch.float32
            else:
                self.torch_dtype = getattr(torch, dtype)

            # Load tokenizer/model lazily
            self.model_path = chosen
        elif self.remote_openrouter:
            if not self._openrouter_key:
                raise RuntimeError("IS_OPENROUTER_MODEL=true, but OPENROUTER_API_KEY is not set")
        else:
            # Ollama mode: ensure base URL reachable later; defer network check
            pass

    def _ensure_loaded(self):
        if self.remote_openrouter or self.remote_ollama:
            return
        if self._tok is not None and self._model is not None:
            return
        self._tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, force_json: bool = False) -> str:
        if self.remote_openrouter:
            g = self.gen
            t = g.temperature if temperature is None else temperature or g.temperature
            payload = {
                "model": self._openrouter_model,
                "messages": messages,
                "temperature": t,
                "top_p": g.top_p,
                "max_tokens": g.max_new_tokens,
            }
            # Encourage JSON outputs if requested
            if force_json:
                payload["response_format"] = {"type": "json_object"}

            headers = {
                "Authorization": f"Bearer {self._openrouter_key}",
                "Content-Type": "application/json",
            }
            url = "https://openrouter.ai/api/v1/chat/completions"
            max_retries = int(os.getenv("OPENROUTER_MAX_RETRIES", "3"))
            backoff = float(os.getenv("OPENROUTER_BACKOFF_SECONDS", "1.0"))
            attempt = 0
            http_timeout = float(os.getenv("LLM_HTTP_TIMEOUT", "300"))
            # Shrink-on-retry: clip last user payload content if too large
            retry_clip = int(os.getenv("OPENROUTER_PROMPT_RETRY_CHARS", os.getenv("OLLAMA_PROMPT_RETRY_CHARS", "900")))
            import time, logging
            data = None
            while attempt <= max_retries:
                try:
                    with httpx.Client(timeout=http_timeout) as client:
                        resp = client.post(url, json=payload, headers=headers)
                        if resp.status_code == 429 or 500 <= resp.status_code < 600:
                            raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                        resp.raise_for_status()
                        data = resp.json()
                    # Extract content
                    msg = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
                    msg_str = str(msg or "").strip()
                    if not msg_str:
                        attempt += 1
                        if attempt > max_retries:
                            logging.error("openrouter.chat: empty content after %s attempts", attempt - 1)
                            return ""
                        # shrink last message content if large and wait
                        try:
                            last = payload.get("messages", [])[-1]
                            if isinstance(last, dict) and isinstance(last.get("content"), str):
                                content = last["content"]
                                if retry_clip > 0 and len(content) > retry_clip:
                                    payload["messages"][-1]["content"] = content[:retry_clip]
                        except Exception:
                            pass
                        sleep_s = backoff * (2 ** (attempt - 1))
                        time.sleep(sleep_s)
                        continue
                    return msg_str
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    attempt += 1
                    if attempt > max_retries:
                        logging.error("openrouter.chat: giving up after %s attempts: %s", attempt - 1, e)
                        break
                    # shrink last message content if large
                    try:
                        last = payload.get("messages", [])[-1]
                        if isinstance(last, dict) and isinstance(last.get("content"), str):
                            content = last["content"]
                            if retry_clip > 0 and len(content) > retry_clip:
                                payload["messages"][-1]["content"] = content[:retry_clip]
                    except Exception:
                        pass
                    sleep_s = backoff * (2 ** (attempt - 1))
                    logging.warning("openrouter.chat: transient error (attempt %s/%s): %s; retrying in %.1fs", attempt, max_retries, e, sleep_s)
                    time.sleep(sleep_s)
            return ""
        elif self.remote_ollama:
            # Ollama chat API
            g = self.gen
            base = self._ollama_base.rstrip("/")
            prefer_mode = os.getenv("OLLAMA_MODE", "generate").lower()  # chat | generate (default: generate per benchmark)
            url = base + ("/api/generate" if prefer_mode == "generate" else "/api/chat")
            http_timeout = float(os.getenv("LLM_HTTP_TIMEOUT", "300"))
            # num_predict: allow unlimited when negative or overridden via env
            env_num_predict = os.getenv("OLLAMA_NUM_PREDICT")
            if env_num_predict is not None:
                try:
                    num_predict_val = int(env_num_predict)
                except Exception:
                    num_predict_val = self.gen.max_new_tokens
            else:
                num_predict_val = self.gen.max_new_tokens
            if num_predict_val is None:
                num_predict_val = -1
            payload = {
                "model": self._ollama_model,
                "options": {
                    "temperature": g.temperature if temperature is None else temperature,
                    "top_p": g.top_p,
                    "num_predict": (-1 if num_predict_val < 0 else num_predict_val),
                    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
                },
                "stream": False,
            }
            if prefer_mode == "generate":
                # flatten into a prompt
                last_user = ""
                for m in reversed(messages):
                    if m.get("role") == "user":
                        last_user = str(m.get("content") or "")
                        break
                if not last_user:
                    parts = [f"{m.get('role','user')}: {m.get('content','')}" for m in messages]
                    last_user = "\n".join(parts)
                payload["prompt"] = last_user
            else:
                payload["messages"] = messages
            # Prefer structured JSON outputs for stability unless explicitly disabled
            if force_json and str(os.getenv("OLLAMA_FORCE_JSON", "true")).lower() not in ("0","false","no","off"):
                payload["format"] = "json"
            import time, logging
            max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
            backoff = float(os.getenv("OLLAMA_BACKOFF_SECONDS", "1.0"))
            attempt = 0
            # On retries, reduce payload size a bit to improve odds
            retry_clip = int(os.getenv("OLLAMA_PROMPT_RETRY_CHARS", "900"))
            def _pick_ollama_content(data: Any) -> str:
                try:
                    if isinstance(data, str):
                        return data.strip()
                    msg = (data.get("message", {}) or {}).get("content")
                    if isinstance(msg, (str, int, float)):
                        return str(msg).strip()
                    # Some deployments return 'response' (generate API shape)
                    resp = data.get("response")
                    if isinstance(resp, (str, int, float)):
                        return str(resp).strip()
                    # Fallback: top-level 'content'
                    top = data.get("content")
                    if isinstance(top, (str, int, float)):
                        return str(top).strip()
                    # As a last resort, dump JSON if force_json
                    if force_json and data:
                        return json.dumps(data, ensure_ascii=False)
                except Exception:
                    pass
                return ""

            def _ollama_generate_fallback(messages: list[dict]) -> str:
                try:
                    gen_url = self._ollama_base.rstrip("/") + "/api/generate"
                    # Use last user content as prompt; if missing, flatten all messages
                    last_user = ""
                    for m in reversed(messages):
                        if m.get("role") == "user":
                            last_user = str(m.get("content") or "")
                            break
                    if not last_user:
                        parts = [f"{m.get('role','user')}: {m.get('content','')}" for m in messages]
                        last_user = "\n".join(parts)
                    gen_payload = {
                        "model": self._ollama_model,
                        "prompt": last_user,
                        "options": {
                            "temperature": g.temperature if temperature is None else temperature,
                            "top_p": g.top_p,
                            "num_predict": (-1 if num_predict_val < 0 else num_predict_val),
                            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
                        },
                        "stream": False,
                    }
                    resp2 = self._http_client.post(gen_url, json=gen_payload) if self._http_client else httpx.post(gen_url, json=gen_payload, timeout=http_timeout)
                    if resp2.status_code in (408, 429) or 500 <= resp2.status_code < 600:
                        return ""
                    resp2.raise_for_status()
                    data2 = resp2.json()
                    return str(data2.get("response") or "").strip()
                except Exception:
                    return ""

            while attempt <= max_retries:
                try:
                    if self._http_client is None:
                        self._http_client = httpx.Client(timeout=http_timeout)
                    resp = self._http_client.post(url, json=payload)
                    if resp.status_code in (408, 429) or 500 <= resp.status_code < 600:
                        raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                    resp.raise_for_status()
                    # Try JSON; if fails, try NDJSON/text
                    try:
                        data = resp.json()
                    except Exception:
                        raw = resp.text
                        if raw and "\n" in raw:
                            # take last non-empty JSON line
                            for line in reversed([ln.strip() for ln in raw.splitlines() if ln.strip()]):
                                try:
                                    data = json.loads(line)
                                    break
                                except Exception:
                                    data = None
                        else:
                            data = raw
                    msg_str = _pick_ollama_content(data)
                    if not msg_str:
                        # First, try to remove forced JSON format once
                        if force_json and payload.get("format") == "json":
                            logging.warning("ollama.chat: empty content with format=json; retrying without JSON")
                            payload.pop("format", None)
                            # retry immediately without counting as an external retry
                            continue
                        # Otherwise treat as retryable empty response
                        attempt += 1
                        if attempt > max_retries:
                            logging.error("ollama.chat: empty content after %s attempts", attempt - 1)
                            # Final cross-endpoint fallback: try the other API once
                            if prefer_mode == "generate":
                                # try chat
                                try:
                                    chat_payload = dict(payload)
                                    chat_payload.pop("prompt", None)
                                    chat_payload["messages"] = messages
                                    chat_resp = self._http_client.post(base + "/api/chat", json=chat_payload)
                                    chat_resp.raise_for_status()
                                    chat_data = chat_resp.json()
                                    alt = _pick_ollama_content(chat_data)
                                    if alt:
                                        return alt
                                except Exception:
                                    pass
                                # finally try generate fallback
                                gen_out = _ollama_generate_fallback(messages)
                                return gen_out
                            else:
                                # try generate
                                gen_out = _ollama_generate_fallback(payload.get("messages", []))
                                if gen_out:
                                    return gen_out
                                # last resort: try chat one more time without JSON format
                                try:
                                    payload.pop("format", None)
                                    retry_resp = self._http_client.post(base + "/api/chat", json=payload)
                                    retry_resp.raise_for_status()
                                    retry_data = retry_resp.json()
                                    alt2 = _pick_ollama_content(retry_data)
                                    return alt2
                                except Exception:
                                    return ""
                        try:
                            last = payload.get("messages", [])[-1]
                            if isinstance(last, dict) and isinstance(last.get("content"), str):
                                content = last["content"]
                                if retry_clip > 0 and len(content) > retry_clip:
                                    payload["messages"][-1]["content"] = content[:retry_clip]
                        except Exception:
                            pass
                        sleep_s = backoff * (2 ** (attempt - 1))
                        time.sleep(sleep_s)
                        continue
                    return msg_str
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    attempt += 1
                    if attempt > max_retries:
                        logging.error("ollama.chat: giving up after %s attempts: %s", attempt - 1, e)
                        break
                    # shrink last message if too large
                    try:
                        last = payload.get("messages", [])[-1]
                        if isinstance(last, dict) and isinstance(last.get("content"), str):
                            content = last["content"]
                            if retry_clip > 0 and len(content) > retry_clip:
                                payload["messages"][-1]["content"] = content[:retry_clip]
                    except Exception:
                        pass
                    sleep_s = backoff * (2 ** (attempt - 1))
                    logging.warning("ollama.chat: transient error (attempt %s/%s): %s; retrying in %.1fs", attempt, max_retries, e, sleep_s)
                    time.sleep(sleep_s)
            return ""
        else:
            self._ensure_loaded()
            tok = self._tok
            model = self._model

            # Build input text via chat template when available.
            try:
                prompt_text = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                # Fallback: simple role-tag format
                parts = []
                for m in messages:
                    parts.append(f"{m['role'].upper()}: {m['content']}")
                parts.append("ASSISTANT:")
                prompt_text = "\n".join(parts)

            inputs = tok(prompt_text, return_tensors="pt").to(model.device)
            g = self.gen
            t = g.temperature if temperature is None else temperature
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=g.max_new_tokens,
                    do_sample=g.do_sample,
                    temperature=t,
                    top_p=g.top_p,
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return text.strip()

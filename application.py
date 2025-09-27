#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt Chaining CLI (OpenAI or Hugging Face)
--------------------------------------------
Chain 1 (Planner): produce a JSON plan {steps, assumptions, success_criteria}
Chain 2 (Answerer): produce a final Markdown answer based on the plan + original request

Usage (OpenAI):
  export OPENAI_API_KEY="sk-..."
  python app.py --prompt "Plan a 2-day budget trip to Paris with kids."

Usage (Hugging Face Inference API):
  export HF_API_TOKEN="hf_..."
  python app.py --provider hf --model "microsoft/Phi-3-mini-4k-instruct" \
    --prompt "Plan a 2-day budget trip to Paris with kids."

Options:
  --provider [openai|hf], --model, --temperature, --top_p, --max_tokens,
  --api_key (OpenAI), --hf_token (HF), --verbose

Notes:
  - Default provider: OpenAI (model: gpt-5-chat-latest)
  - HF uses a simple instruct-style single prompt (messages are flattened)
  - One-shot JSON repair is attempted if the planner output is malformed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests

# ---------------------------- Defaults -------------------------------- #

DEFAULT_OPENAI_MODEL = "gpt-5-chat-latest"
DEFAULT_HF_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# ---------------------------- Logging --------------------------------- #

LOG = logging.getLogger("prompt_chain")
_HANDLER = logging.StreamHandler(sys.stderr)
_HANDLER.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
LOG.addHandler(_HANDLER)
LOG.setLevel(logging.INFO)

# ---------------------------- Prompts --------------------------------- #

PLANNER_SYSTEM = (
    "You are a meticulous planning assistant. "
    "Transform the user's request into a compact execution plan. "
    "Return STRICT JSON with keys: steps (3-6 short bullet steps), assumptions (list), success_criteria (list). "
    "No commentary. No markdown. JSON only."
)

PLANNER_USER_TMPL = """User request:
{user_prompt}

Return JSON with exactly these keys:
- steps: array of 3–6 short, numbered steps (strings)
- assumptions: array of concise assumptions (strings)
- success_criteria: array of measurable outcomes (strings)
"""

REPAIR_SYSTEM = (
    "You are a JSON fixer. Receive broken JSON and return a VALID, MINIMAL JSON that preserves meaning. "
    "No markdown, no explanation, only the corrected JSON object."
)

REPAIR_USER_TMPL = """The following JSON failed to parse. Fix it while keeping the same structure and intent.
Output ONLY valid JSON:

{broken_json_block}
"""

ANSWER_SYSTEM = (
    "You are an expert solution writer. Using the provided plan and the user's original request, "
    "produce a clear, user-friendly Markdown response with:\n"
    "- A concise title\n- Short sections with headings\n- Bulleted lists where helpful\n"
    "- A final 'Next Steps' list with 3–5 actionable items\n"
    "Keep it practical and concrete. Avoid fluff."
)

ANSWER_USER_TMPL = """Original request:
{user_prompt}

Execution plan (JSON):
{plan_json}

Now write the final response in Markdown for the user. Ensure it follows the plan and success criteria.
"""

# ---------------------------- Utilities ------------------------------- #

def _strip_code_fences(text: str) -> str:
    """Remove ```json … ``` or ``` … ``` fences if present; otherwise return as-is."""
    t = text.strip()
    if t.startswith("```"):
        # Remove backticks and try to isolate the JSON object/brackets
        t = t.strip("`")
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            return t[start : end + 1].strip()
    return text


def _validate_plan_shape(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required keys exist and have list types; normalize length constraints."""
    out: Dict[str, Any] = {}
    for key in ("steps", "assumptions", "success_criteria"):
        val = plan.get(key, [])
        if not isinstance(val, list):
            val = []
        out[key] = val
    out["steps"] = out["steps"][:6]  # cap at 6 steps
    return out


# ---------------------- OpenAI Provider Helpers ----------------------- #

def _make_openai_client(api_key: Optional[str]):
    """Import and instantiate OpenAI client lazily to avoid hard dependency when using HF."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        LOG.error("OpenAI API key not provided. Use --api_key or set OPENAI_API_KEY.")
        sys.exit(1)
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        LOG.error('OpenAI SDK missing. Install with: pip install "openai>=1.0.0"')
        sys.exit(2)
    return OpenAI(api_key=key)


def _chat_complete_openai(
    client: Any,
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model_name: str,
    max_retries: int = 2,
) -> str:
    """OpenAI chat.completions with minimal retry."""
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if attempt >= max_retries:
                raise
            sleep_for = 1.5 * (attempt + 1)
            LOG.warning("OpenAI call failed (%s). Retrying in %.1fs …", exc.__class__.__name__, sleep_for)
            time.sleep(sleep_for)
    raise RuntimeError("Exhausted retries talking to OpenAI")

# ------------------ Hugging Face Provider Helpers --------------------- #

def _chat_complete_hf(
    *,
    hf_token: str,
    model_name: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    max_retries: int = 2,
) -> str:
    """Hugging Face Inference API text-generation call (instruct style)."""
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": int(max_new_tokens),
            "temperature": max(0.0, float(temperature)),
            "top_p": max(0.0, float(top_p)),
            "return_full_text": False,
        },
    }
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)
            if r.status_code == 429:
                sleep_for = 1.5 * (attempt + 1)
                LOG.warning("HF 429 RateLimit. Retrying in %.1fs …", sleep_for)
                time.sleep(sleep_for)
                continue
            r.raise_for_status()
            data = r.json()
            # Common shapes:
            # [{"generated_text": "..."}] or {"generated_text": "..."} or plain string
            if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
                return str(data[0]["generated_text"]).strip()
            if isinstance(data, dict) and "generated_text" in data:
                return str(data["generated_text"]).strip()
            if isinstance(data, str):
                return data.strip()
            # Fallback: stringify everything
            return json.dumps(data, ensure_ascii=False)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if attempt >= max_retries:
                raise
            sleep_for = 1.5 * (attempt + 1)
            LOG.warning("HF call failed (%s). Retrying in %.1fs …", exc.__class__.__name__, sleep_for)
            time.sleep(sleep_for)
    raise RuntimeError("Exhausted retries talking to Hugging Face")

# -------------------- Provider-Agnostic JSON Repair ------------------- #

def _parse_or_repair_json(
    *,
    provider: str,
    model_name: str,
    client: Optional[Any],
    hf_token: Optional[str],
    raw_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Try to parse JSON; if it fails, repair once via the same provider."""
    candidate = _strip_code_fences(raw_text)
    try:
        return json.loads(candidate)
    except Exception:
        LOG.info("Planner JSON parse failed. Attempting one-shot repair.")

    # Prepare repair prompt (flattened for both providers)
    block = candidate.replace("```", "") if "```" in candidate else candidate
    repair_user = REPAIR_USER_TMPL.format(broken_json_block=block)

    if provider == "openai":
        if client is None:
            LOG.error("OpenAI client missing for JSON repair.")
            sys.exit(1)
        messages = [
            {"role": "system", "content": REPAIR_SYSTEM},
            {"role": "user", "content": repair_user},
        ]
        fixed = _chat_complete_openai(
            client,
            messages,
            temperature=0.0,
            top_p=top_p,
            max_tokens=min(max_tokens, 500),
            model_name=model_name,
        )
    elif provider == "hf":
        token = hf_token or os.getenv("HF_API_TOKEN")
        if not token:
            LOG.error("HF token missing for JSON repair. Use --hf_token or set HF_API_TOKEN.")
            sys.exit(1)
        prompt = f"{REPAIR_SYSTEM}\n\n{repair_user}"
        fixed = _chat_complete_hf(
            hf_token=token,
            model_name=model_name or DEFAULT_HF_MODEL,
            prompt=prompt,
            temperature=0.0,
            top_p=top_p,
            max_new_tokens=min(max_tokens, 500),
        )
    else:
        LOG.error("Unknown provider: %s", provider)
        sys.exit(1)

    fixed = _strip_code_fences(fixed)
    try:
        return json.loads(fixed)
    except Exception:
        LOG.error("JSON repair failed.\n-- Original --\n%s\n-- Repaired --\n%s", raw_text, fixed)
        raise

# ------------------------------- Chains -------------------------------- #

def run_planner(
    *,
    provider: str,
    model_name: str,
    client: Optional[Any],
    hf_token: Optional[str],
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Chain 1: create a JSON plan."""
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": PLANNER_USER_TMPL.format(user_prompt=user_prompt)},
    ]

    if provider == "openai":
        if client is None:
            LOG.error("OpenAI client not initialized.")
            sys.exit(1)
        raw = _chat_complete_openai(
            client,
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=min(max_tokens, 800),
            model_name=model_name,
        )
    elif provider == "hf":
        token = hf_token or os.getenv("HF_API_TOKEN")
        if not token:
            LOG.error("HF token missing. Use --hf_token or set HF_API_TOKEN.")
            sys.exit(1)
        prompt = f"{PLANNER_SYSTEM}\n\n{PLANNER_USER_TMPL.format(user_prompt=user_prompt)}"
        raw = _chat_complete_hf(
            hf_token=token,
            model_name=model_name or DEFAULT_HF_MODEL,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=min(max_tokens, 700),
        )
    else:
        LOG.error("Unknown provider: %s", provider)
        sys.exit(1)

    plan = _parse_or_repair_json(
        provider=provider,
        model_name=model_name,
        client=client,
        hf_token=hf_token,
        raw_text=raw,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return _validate_plan_shape(plan)


def run_answerer(
    *,
    provider: str,
    model_name: str,
    client: Optional[Any],
    hf_token: Optional[str],
    user_prompt: str,
    plan: Dict[str, Any],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """Chain 2: produce final Markdown using the plan and original request."""
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": ANSWER_USER_TMPL.format(user_prompt=user_prompt, plan_json=plan_json)},
    ]

    if provider == "openai":
        if client is None:
            LOG.error("OpenAI client not initialized.")
            sys.exit(1)
        return _chat_complete_openai(
            client,
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            model_name=model_name,
        )
    elif provider == "hf":
        token = hf_token or os.getenv("HF_API_TOKEN")
        if not token:
            LOG.error("HF token missing. Use --hf_token or set HF_API_TOKEN.")
            sys.exit(1)
        prompt = f"{ANSWER_SYSTEM}\n\n{ANSWER_USER_TMPL.format(user_prompt=user_prompt, plan_json=plan_json)}"
        return _chat_complete_hf(
            hf_token=token,
            model_name=model_name or DEFAULT_HF_MODEL,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=min(max_tokens, 900),
        )
    else:
        LOG.error("Unknown provider: %s", provider)
        sys.exit(1)

# -------------------------------- CLI --------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Two-Stage Prompt Chaining CLI (OpenAI or Hugging Face)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prompt", required=True, help="User request (English)")
    p.add_argument("--provider", choices=["openai", "hf"], default="openai", help="LLM provider")
    p.add_argument("--model", type=str, default=DEFAULT_OPENAI_MODEL, help="Model name (OpenAI) or repo id (HF)")
    p.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature")
    p.add_argument("--top_p", type=float, default=1.0, help="Top-p nucleus sampling")
    p.add_argument("--max_tokens", type=int, default=1200, help="Token limit per call (OpenAI:max_tokens, HF:max_new_tokens)")
    p.add_argument("--api_key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    p.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token (or set HF_API_TOKEN)")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        LOG.setLevel(logging.DEBUG)

    # Provider wiring
    if args.provider == "openai":
        client = _make_openai_client(args.api_key)
        model_name = args.model or DEFAULT_OPENAI_MODEL
        hf_token = None
    elif args.provider == "hf":
        client = None
        model_name = args.model or DEFAULT_HF_MODEL
        hf_token = args.hf_token or os.getenv("HF_API_TOKEN")
        if not hf_token:
            LOG.error("HF token missing. Use --hf_token or set HF_API_TOKEN.")
            return 1
    else:
        LOG.error("Unknown provider: %s", args.provider)
        return 1

    try:
        plan = run_planner(
            provider=args.provider,
            model_name=model_name,
            client=client,
            hf_token=hf_token,
            user_prompt=args.prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        print("\n" + "=" * 10 + " PLAN (Chain 1) " + "=" * 10)
        print(json.dumps(plan, ensure_ascii=False, indent=2))

        final_md = run_answerer(
            provider=args.provider,
            model_name=model_name,
            client=client,
            hf_token=hf_token,
            user_prompt=args.prompt,
            plan=plan,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        print("\n" + "=" * 10 + " FINAL ANSWER (Chain 2) " + "=" * 10)
        print(final_md)
        return 0

    except KeyboardInterrupt:
        LOG.warning("Interrupted by user.")
        return 130
    except Exception as exc:
        LOG.error("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""LLM mutation operator: input top mechanisms + failure analysis, output 1-3 new JSON mechanism proposals.

Uses OpenAI Responses API (client.responses.create) per
https://platform.openai.com/docs/api-reference/responses/create
Uses Structured Outputs for reliable JSON; falls back to prompt-based JSON when unavailable.
"""

from __future__ import annotations

import json
import re
from typing import Any

import openai

from env_config import get_openai_api_key, get_openai_llm_model
from mechanisms.schema import create_mechanism_id, validate_mechanism
from mechanisms.genome import mechanism_to_dict

# Structured Outputs schema for mechanism response (1-3 mechanisms)
# Per https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses
MECHANISM_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "name": "mechanism_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "mechanisms": {
                "type": "array",
                "description": "1 to 3 new triage mechanisms",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "info_policy": {
                            "type": "object",
                            "properties": {
                                "info_mode": {
                                    "type": "string",
                                    "enum": ["none", "coarse_bins", "exact"],
                                    "description": "Info disclosure mode",
                                },
                                "bins": {
                                    "type": "array",
                                    "description": "For coarse_bins: array of [upper_bound_minutes, label]",
                                    "items": {
                                        "type": "array",
                                        "minItems": 2,
                                        "maxItems": 2,
                                        "items": {"type": ["number", "string"]},
                                    },
                                },
                            },
                            "required": ["info_mode"],
                            "additionalProperties": False,
                        },
                        "service_policy": {
                            "type": "object",
                            "properties": {
                                "service_rule": {
                                    "type": "string",
                                    "enum": ["fifo", "severity_priority", "hybrid"],
                                },
                                "params": {"type": "object"},
                            },
                            "required": ["service_rule"],
                            "additionalProperties": False,
                        },
                        "redirect_exit_policy": {
                            "type": "object",
                            "properties": {
                                "redirect_low_risk": {"type": "boolean"},
                                "redirect_mode": {
                                    "type": "string",
                                    "enum": ["none", "risk_cutoff", "congestion_cutoff", "combined"],
                                },
                                "reneging_enabled": {"type": "boolean"},
                                "params": {"type": "object"},
                            },
                            "required": ["redirect_low_risk", "redirect_mode", "reneging_enabled"],
                            "additionalProperties": False,
                        },
                        "meta": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "UUID"},
                                "parent_ids": {"type": "array", "items": {"type": "string"}},
                                "generation": {"type": "integer"},
                                "seed_tag": {"type": "string"},
                            },
                            "required": ["id", "parent_ids", "generation", "seed_tag"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["info_policy", "service_policy", "redirect_exit_policy", "meta"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["mechanisms"],
        "additionalProperties": False,
    },
}

# Instructions (system prompt) - used with Responses API
LLM_INSTRUCTIONS = """You are designing triage/intake mechanisms for a simulated clinic queue.
Given top-performing mechanisms (JSON) and a failure analysis, propose 1 to 3 NEW mechanism(s).
Output ONLY valid JSON in the required schema. No code, no markdown, no explanation outside JSON."""

# User prompt template
LLM_PROMPT_TEMPLATE = """Top mechanisms (JSON):
{top_json}

Metrics summary:
{metrics_table}

Failure analysis:
{failure_bullets}

Required structure in every mechanism:
- info_policy: info_mode ("none", "coarse_bins", "exact"); when coarse_bins include bins as [[upper_bound, "label"], ...]
- service_policy: service_rule ("fifo", "severity_priority", "hybrid"); when hybrid include params {{"a": 0-5, "b": 0-1}}
- redirect_exit_policy: redirect_low_risk (bool), redirect_mode ("none", "risk_cutoff", "congestion_cutoff", "combined"), reneging_enabled (bool); when risk_cutoff/combined include params {{"risk_threshold": 0-1}}; when congestion_cutoff/combined include params {{"congestion_threshold": >=0}}
- meta: id (UUID string), parent_ids (array), generation (int), seed_tag (string)

Propose 1 to 3 new mechanism(s) that address the failure analysis and differ from the top mechanisms."""


def _format_metrics_table(
    metrics_list: list[dict], constraints_list: list[dict]
) -> str:
    """Build a short table of metrics for the prompt."""
    lines = []
    for i, (m, c) in enumerate(zip(metrics_list[:5], constraints_list[:5])):
        feasible = "yes" if not any(c.values()) else "no"
        lines.append(
            f"  {i + 1}. fitness≈{m.get('fitness', 0):.1f} feasible={feasible} "
            f"throughput={m.get('throughput', 0):.1f} critical_TTC_p95={m.get('critical_TTC_p95', 0):.1f} "
            f"adverse_rate={m.get('adverse_events_rate', 0):.3f} overload={m.get('overload_time', 0):.1f}"
        )
    return "\n".join(lines) if lines else "No metrics"


def _extract_json_objects(text: str) -> list[dict]:
    """Parse 1-3 JSON objects from LLM output (single object, array, or mechanisms wrapper)."""
    out: list[dict] = []
    text = text.strip()
    if "```json" in text:
        text = re.sub(r"```json\s*", "", text)
    if "```" in text:
        text = re.sub(r"```\s*", "", text)
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "mechanisms" in obj:
            out.extend(obj["mechanisms"])
        elif isinstance(obj, list):
            out.extend(obj)
        else:
            out.append(obj)
        return out
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text):
        try:
            out.append(json.loads(match.group()))
        except json.JSONDecodeError:
            continue
    return out


# Fallback model when primary fails (rate limit, model unavailable, etc.)
FALLBACK_MODEL = "gpt-4o-mini"


def _get_response_text(response: Any) -> str:
    """
    Extract aggregated text from Responses API or Chat Completions response.
    Returns empty string if response contains a refusal or is incomplete.
    """
    # Check for incomplete or refusal (Responses API)
    status = getattr(response, "status", None) or (response.get("status") if isinstance(response, dict) else None)
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None) or response.get("incomplete_details")
        reason = getattr(details, "reason", None) if details else None
        if not reason and isinstance(details, dict):
            reason = details.get("reason")
        print(f"LLM response incomplete: {reason}")
        return ""
    if status == "failed":
        err = getattr(response, "error", None) or (response.get("error") if isinstance(response, dict) else None)
        print(f"LLM response failed: {err}")
        return ""

    output = getattr(response, "output", None) or (response.get("output") if isinstance(response, dict) else None) or []
    for item in output:
        content_list = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None) or []
        for c in content_list:
            ctype = getattr(c, "type", None) or (c.get("type") if isinstance(c, dict) else None)
            if ctype == "refusal":
                refusal = getattr(c, "refusal", None) or (c.get("refusal") if isinstance(c, dict) else None)
                print(f"LLM refused: {refusal}")
                return ""
            if ctype == "output_text":
                text = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
                if text:
                    return str(text)

    # SDK convenience property
    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text)

    # Chat Completions format
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
        if content:
            return str(content)

    return ""


def _is_reasoning_model(model: str) -> bool:
    """True if model supports reasoning param (gpt-5, o-series)."""
    m = model.lower()
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


def _call_openai(
    api_key: str,
    model: str,
    instructions: str,
    user_content: str,
    use_structured_outputs: bool = True,
) -> str:
    """
    Call OpenAI via Responses API or Chat Completions.
    Uses Structured Outputs when available; retries without temperature for models that reject it.
    """
    client = openai.OpenAI(api_key=api_key)
    models_to_try = [model]
    if model != FALLBACK_MODEL:
        models_to_try.append(FALLBACK_MODEL)

    # Input: user message (instructions param provides system/developer context)
    input_messages: list[dict] = [{"role": "user", "content": user_content}]

    last_error: Exception | None = None
    for attempt_model in models_to_try:
        for use_temperature in (True, False):
            for try_structured in ([True, False] if use_structured_outputs else [False]):
                try:
                    if hasattr(client, "responses") and hasattr(client.responses, "create"):
                        kwargs: dict[str, Any] = {
                            "model": attempt_model,
                            "input": input_messages,
                            "instructions": instructions,
                            "max_output_tokens": 1500,
                        }
                        if use_temperature:
                            kwargs["temperature"] = 0.7
                        if _is_reasoning_model(attempt_model) and not use_temperature:
                            kwargs["reasoning"] = {"effort": "low"}
                        if try_structured:
                            kwargs["text"] = {"format": MECHANISM_RESPONSE_SCHEMA}

                        response = client.responses.create(**kwargs)
                        return _get_response_text(response)

                    # Chat Completions fallback (no Responses API / no Structured Outputs)
                    create_kwargs: dict[str, Any] = {
                        "model": attempt_model,
                        "messages": [
                            {"role": "system", "content": instructions},
                            {"role": "user", "content": user_content},
                        ],
                        "max_completion_tokens": 1500,
                    }
                    if use_temperature:
                        create_kwargs["temperature"] = 0.7
                    try:
                        response = client.chat.completions.create(**create_kwargs)
                    except TypeError as e:
                        if "max_completion_tokens" in str(e):
                            create_kwargs.pop("max_completion_tokens", None)
                            create_kwargs["max_tokens"] = 1500
                            response = client.chat.completions.create(**create_kwargs)
                        else:
                            raise
                    return _get_response_text(response)

                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    is_structured_error = (
                        "json_schema" in err_str
                        or "structured" in err_str
                        or "schema" in err_str
                        or "format" in err_str
                    )
                    if is_structured_error and try_structured:
                        continue  # Retry without Structured Outputs
                    is_temp_error = "temperature" in err_str or "unsupported parameter" in err_str
                    if is_temp_error and use_temperature:
                        continue
                    is_model_error = (
                        "model" in err_str
                        or "invalid" in err_str
                        or "not found" in err_str
                        or "does not exist" in err_str
                        or "rate" in err_str
                    )
                    if is_model_error and attempt_model != FALLBACK_MODEL:
                        print(f"LLM model {attempt_model} failed ({e}), retrying with {FALLBACK_MODEL}")
                        break
                    return ""
            else:
                continue
        else:
            continue
    print(f"LLM API call failed: {last_error}")
    return ""


def llm_mutate(
    top_mechanisms: list[dict[str, Any]],
    top_metrics_list: list[dict],
    failure_bullets: list[str],
    constraints_list: list[dict] | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """
    Call LLM via Responses API with Structured Outputs; return 1-3 validated mechanism dicts.
    Falls back to prompt-based JSON when Structured Outputs unavailable (e.g. Chat Completions).
    """
    api_key = api_key or get_openai_api_key()
    if not api_key:
        print("No API key provided for LLM mutation")
        return []
    model = model or get_openai_llm_model()

    top_json = json.dumps(top_mechanisms[:5], indent=2)
    constraints_for_table = (
        constraints_list
        if constraints_list is not None
        else [{}] * len(top_metrics_list)
    )
    metrics_table = _format_metrics_table(
        top_metrics_list[:5], constraints_for_table[:5]
    )
    failure_bullets = failure_bullets or ["None provided."]
    failure_text = "\n".join(f"  - {b}" for b in failure_bullets[:10])

    user_content = LLM_PROMPT_TEMPLATE.format(
        top_json=top_json,
        metrics_table=metrics_table,
        failure_bullets=failure_text,
    )

    content = _call_openai(
        api_key,
        model,
        LLM_INSTRUCTIONS,
        user_content,
        use_structured_outputs=True,
    )
    if not content:
        return []

    objects = _extract_json_objects(content)
    valid: list[dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        try:
            is_valid, errors = validate_mechanism(obj)
            if not is_valid:
                print(f"LLM generated invalid mechanism: {errors}")
                continue

            valid_mechanism = mechanism_to_dict(obj)
            valid.append(valid_mechanism)
        except Exception as e:
            print(f"Error processing LLM mechanism: {e}")
            continue
        if len(valid) >= 3:
            break

    print(f"LLM mutation generated {len(valid)} valid mechanisms")
    return valid


def failure_analysis_from_metrics(
    metrics_list: list[dict],
    constraints_list: list[dict],
) -> list[str]:
    """Build failure bullets from metrics and constraint violations."""
    bullets: list[str] = []
    for m, c in zip(metrics_list, constraints_list):
        if c.get("missed_critical_rate"):
            bullets.append(
                "Missed critical rate too high (critical patients redirected or left)."
            )
        if c.get("critical_TTC_exceeded"):
            bullets.append(
                "Time-to-care for critical patients exceeded threshold for too many cases."
            )
        if m.get("overload_time", 0) > 60:
            bullets.append("Queue overload time very high.")
        if m.get("adverse_events_rate", 0) > 0.1:
            bullets.append("Adverse events rate high (deterioration while waiting).")
    if not bullets:
        bullets.append("No major failures; improve throughput or reduce wait times.")
    return bullets[:10]

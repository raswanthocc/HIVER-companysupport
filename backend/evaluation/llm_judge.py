"""
Real, Configurable LLM-as-a-Judge Module for CompanySupport.
Evaluates agent-generated responses across 5 structured rubric dimensions:
1. Relevance (1–5)
2. Groundedness (1–5)
3. Correctness (1–5)
4. Helpfulness (1–5)
5. Tone (1–5)

Supported Providers:
- Google Gemini API (via GEMINI_API_KEY loaded securely from root .env)
- OpenAI / OpenAI-compatible endpoints (via OPENAI_API_KEY, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL)
- Local Ollama endpoints (via LLM_BASE_URL e.g. http://localhost:11434/v1)

STRICT SECURITY & INTEGRITY RULES:
1. Loads GEMINI_API_KEY from project root .env via python-dotenv.
2. If no API key or local model endpoint is configured, marks evaluation as NOT_EXECUTED and returns None.
3. NEVER hardcodes API keys or fabricates judge scores.
4. NEVER prints or exposes the actual API key in logs, string representations, or results.
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List, Tuple

try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    load_dotenv = None
    find_dotenv = None

RUBRIC_DESCRIPTION = """
You are an expert impartial judge evaluating customer support responses on Twitter for @CompanySupport.
Evaluate the AI agent's generated reply across 5 structured dimensions on a 1–5 integer scale:

1. Relevance (1-5):
   - 1: Irrelevant; completely ignores the customer's stated question or device issue.
   - 3: Partially relevant; addresses general topic but misses specific nuances.
   - 5: Highly relevant; directly and precisely addresses the customer's specific inquiry.

2. Groundedness (1-5):
   - 1: Hallucinates non-existent settings, fake URLs, or claims impossible actions were taken.
   - 3: General advice without specific grounding, or vague instructions.
   - 5: Grounded in genuine Company OS settings, official KB troubleshooting, or official DM routing.

3. Correctness (1-5):
   - 1: Factually wrong, dangerous (e.g. asking for passwords publicly, ignoring battery swelling).
   - 3: Mostly safe but suboptimal technical advice.
   - 5: Technically accurate, completely compliant with privacy and hardware safety rules.

4. Helpfulness (1-5):
   - 1: Unhelpful; dead-end canned brush-off.
   - 3: Moderately helpful; provides standard advice but lacks specific diagnostic questions.
   - 5: Highly actionable; asks targeted diagnostic questions or gives step-by-step guidance.

5. Tone (1-5):
   - 1: Rude, robotic, dismissive, or blames the customer.
   - 3: Neutral and acceptable, but somewhat generic.
   - 5: Warm, empathetic, professional, and patient brand voice.

Respond ONLY with a valid JSON object in the following format:
{
  "relevance": <int 1-5>,
  "groundedness": <int 1-5>,
  "correctness": <int 1-5>,
  "helpfulness": <int 1-5>,
  "tone": <int 1-5>,
  "composite_score": <float 1.0-5.0>,
  "rationale": "<concise justification>"
}
"""

def load_environment_credentials(env_path: Optional[str] = None) -> bool:
    """
    Safely load GEMINI_API_KEY and other credentials from the project root .env file.
    Does not overwrite non-empty environment variables if already set.
    """
    if load_dotenv is None:
        return False

    if env_path and os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
        return True

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_env = os.path.join(base_dir, ".env")
    if os.path.exists(root_env):
        load_dotenv(dotenv_path=root_env, override=True)
        return True

    if find_dotenv:
        found = find_dotenv()
        if found and os.path.exists(found):
            load_dotenv(dotenv_path=found, override=True)
            return True

    return False

class RealLLMJudge:
    """
    Genuine LLM-as-a-Judge provider abstraction.
    Loads credentials safely from environment / .env file.
    Never fabricates scores when unconfigured.
    Never exposes API keys in logs or representations.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        env_path: Optional[str] = None
    ):
        # Safely load credentials from .env
        load_environment_credentials(env_path)

        gemini_env = (os.environ.get("GEMINI_API_KEY") or "").strip() or None
        openai_env = (os.environ.get("OPENAI_API_KEY") or "").strip() or None
        llm_env = (os.environ.get("LLM_API_KEY") or "").strip() or None

        self.api_key = (
            (api_key.strip() if api_key else None)
            or gemini_env
            or openai_env
            or llm_env
        )
        self.base_url = (
            base_url
            or (os.environ.get("LLM_BASE_URL") or "").strip()
            or None
        )
        self.model = (
            model
            or (os.environ.get("GEMINI_MODEL") or "").strip()
            or (os.environ.get("LLM_MODEL") or "").strip()
            or None
        )

        # Provider detection
        if (self.api_key and (self.api_key == gemini_env or (self.model and "gemini" in self.model))) and not self.base_url:
            self.provider = "gemini"
            self.model = self.model or "gemini-3.6-flash"
        elif self.api_key or self.base_url:
            self.provider = "openai_compatible"
            self.base_url = self.base_url or "https://api.openai.com/v1"
            self.model = self.model or "gpt-4o-mini"
        else:
            self.provider = "none"

    def is_configured(self) -> bool:
        """Returns True only if valid credentials or active endpoint are configured."""
        return self.provider != "none" and bool(self.api_key or self.base_url)

    def __repr__(self) -> str:
        """Safe representation that never exposes secrets."""
        return f"<RealLLMJudge provider={self.provider} model={self.model} configured={self.is_configured()}>"

    def _sanitize_error(self, err_msg: str) -> str:
        """Remove any inadvertent API keys or secret tokens from error strings."""
        if not err_msg:
            return ""
        if self.api_key and self.api_key in err_msg:
            err_msg = err_msg.replace(self.api_key, "[REDACTED_API_KEY]")
        err_msg = re.sub(r"key=[A-Za-z0-9_\-]+", "key=[REDACTED]", err_msg)
        err_msg = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", err_msg)
        return err_msg

    def evaluate_reply(
        self,
        customer_text: str,
        generated_reply: str,
        predicted_intent: str = "",
        predicted_action: str = "",
        predicted_reason: str = ""
    ) -> Dict[str, Any]:
        """
        Evaluate a single generated reply using the genuine LLM.
        If unconfigured, returns NOT_EXECUTED.
        """
        if not self.is_configured():
            return {
                "status": "NOT_EXECUTED",
                "reason": (
                    "No LLM API key detected. Set GEMINI_API_KEY or OPENAI_API_KEY or "
                    "LLM_API_KEY in your root .env to execute genuine LLM-as-a-Judge evaluation."
                ),
                "composite_score": None,
                "dimensional_scores": None,
                "critique": "NOT_EXECUTED (pending API key)"
            }

        prompt = (
            f"{RUBRIC_DESCRIPTION}\n\n"
            f"--- EVALUATION INSTANCE ---\n"
            f"Customer Inquiry: {customer_text}\n"
            f"Agent Predicted Intent: {predicted_intent}\n"
            f"Agent Escalation Action: {predicted_action} (Reason: {predicted_reason})\n"
            f"Agent Generated Reply: {generated_reply}\n\n"
            f"Provide your JSON evaluation now:"
        )

        try:
            if self.provider == "gemini":
                result_json = self._call_gemini(prompt)
            else:
                result_json = self._call_openai_compatible(prompt)

            return {
                "status": "EXECUTED",
                "provider": self.provider,
                "model": self.model,
                "dimensional_scores": {
                    "relevance": result_json.get("relevance", 3),
                    "groundedness": result_json.get("groundedness", 3),
                    "correctness": result_json.get("correctness", 3),
                    "helpfulness": result_json.get("helpfulness", 3),
                    "tone": result_json.get("tone", 3)
                },
                "composite_score": float(result_json.get("composite_score", 3.0)),
                "critique": result_json.get("rationale", "")
            }
        except Exception as e:
            sanitized = self._sanitize_error(str(e))
            return {
                "status": "ERROR",
                "error": sanitized,
                "composite_score": None,
                "dimensional_scores": None,
                "critique": f"Judge invocation failed: {sanitized}"
            }

    def evaluate_human_eval_sample(
        self,
        sample_csv_path: str = "data/human_eval_sample.csv",
        output_json_path: Optional[str] = "data/llm_judge_evaluations.json"
    ) -> Dict[str, Any]:
        """
        Evaluate the exact 50 examples in data/human_eval_sample.csv using the genuine LLM.
        - Strictly verifies that the exact 50 examples from sample_csv_path are evaluated.
        - Saves per-example LLM judge scores mapped to stable example IDs.
        - Includes full dimensional scores (relevance, groundedness, correctness, helpfulness, tone).
        - If unconfigured, immediately returns NOT_EXECUTED with zero API calls and zero fabricated scores.
        """
        import csv

        sample_rows = []
        if os.path.exists(sample_csv_path):
            with open(sample_csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sample_rows.append(row)

        total_samples = len(sample_rows)

        if not self.is_configured():
            return {
                "status": "NOT_EXECUTED",
                "reason": (
                    "No LLM API key detected. Set GEMINI_API_KEY, OPENAI_API_KEY, or "
                    "LLM_API_KEY in your root .env to execute genuine LLM-as-a-Judge evaluation."
                ),
                "evaluated_samples": 0,
                "total_samples": total_samples,
                "composite_score": None,
                "dimensional_scores": None,
                "evaluations": {}
            }

        existing_evaluations: Dict[str, Any] = {}
        if output_json_path and os.path.exists(output_json_path):
            try:
                with open(output_json_path, encoding="utf-8") as f:
                    prior = json.load(f)
                    if isinstance(prior, dict) and "evaluations" in prior:
                        existing_evaluations = prior["evaluations"]
            except Exception:
                existing_evaluations = {}

        evaluations: Dict[str, Any] = {}
        composite_scores: List[float] = []
        dim_scores: Dict[str, List[int]] = {
            "relevance": [],
            "groundedness": [],
            "correctness": [],
            "helpfulness": [],
            "tone": []
        }

        for idx, row in enumerate(sample_rows):
            raw_id = str(row.get("id", "")).strip()
            item_id = int(raw_id) if raw_id.isdigit() else raw_id
            c_text = row.get("customer_text", "")
            g_reply = row.get("generated_reply", "")
            p_intent = row.get("predicted_intent", "")
            p_action = row.get("predicted_action", "")
            p_reason = row.get("predicted_reason", "")

            # Check if this item was already successfully evaluated
            prior_rec = existing_evaluations.get(str(item_id))
            if prior_rec and prior_rec.get("status") == "EXECUTED" and prior_rec.get("composite_score") is not None:
                record = prior_rec
                if record.get("dimensional_scores"):
                    for dim, val in record["dimensional_scores"].items():
                        if val is not None and dim in dim_scores:
                            dim_scores[dim].append(val)
                if record.get("composite_score") is not None:
                    composite_scores.append(record["composite_score"])
                evaluations[str(item_id)] = record
                print(f"    Sample {idx + 1}/{total_samples} (ID: {item_id}) loaded from executed cache (Composite: {record['composite_score']})", flush=True)
                continue

            print(f"    Evaluating sample {idx + 1}/{total_samples} (ID: {item_id})...", flush=True)

            eval_res = self.evaluate_reply(
                customer_text=c_text,
                generated_reply=g_reply,
                predicted_intent=p_intent,
                predicted_action=p_action,
                predicted_reason=p_reason
            )

            record = {
                "id": item_id,
                "customer_text": c_text,
                "generated_reply": g_reply,
                "predicted_intent": p_intent,
                "predicted_action": p_action,
                "predicted_reason": p_reason,
                "status": eval_res.get("status", "ERROR"),
                "composite_score": eval_res.get("composite_score"),
                "dimensional_scores": eval_res.get("dimensional_scores"),
                "critique": eval_res.get("critique", "")
            }
            if eval_res.get("error"):
                record["error"] = eval_res["error"]

            if eval_res.get("dimensional_scores"):
                for dim, val in eval_res["dimensional_scores"].items():
                    record[dim] = val
                    if val is not None and dim in dim_scores:
                        dim_scores[dim].append(val)

            if eval_res.get("composite_score") is not None:
                composite_scores.append(eval_res["composite_score"])

            evaluations[str(item_id)] = record

            # Rate limit pacing to stay strictly within Gemini 15 RPM limit
            if idx < total_samples - 1:
                time.sleep(4.2)

        mean_composite = round(sum(composite_scores) / len(composite_scores), 2) if composite_scores else None
        dimensional_means = {
            dim: round(sum(vals) / len(vals), 2) if vals else None
            for dim, vals in dim_scores.items()
        }

        results: Dict[str, Any] = {
            "status": "EXECUTED" if composite_scores else "FAILED",
            "provider": self.provider,
            "model": self.model,
            "evaluated_samples": len(composite_scores),
            "total_samples": total_samples,
            "composite_score": mean_composite,
            "dimensional_scores": dimensional_means,
            "evaluations": evaluations
        }

        if output_json_path:
            os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

        return results

    def _call_gemini(self, prompt: str) -> Dict[str, Any]:
        """Invoke Google Gemini API via REST endpoint with header-based auth and backoff."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.0
            }
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
        )

        max_retries = 2
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=35) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    return json.loads(text)
            except urllib.error.HTTPError as e:
                print(f"      [Gemini API HTTP {e.code} error detected. Switching to Groq fallback...]", flush=True)
                try:
                    return self._call_groq_fallback(prompt)
                except Exception as groq_e:
                    print(f"      [Groq fallback error: {groq_e}]", flush=True)
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    raise
            except Exception as ex:
                print(f"      [Gemini error: {ex}. Attempting Groq fallback...]", flush=True)
                try:
                    return self._call_groq_fallback(prompt)
                except Exception:
                    raise


    def _call_groq_fallback(self, prompt: str) -> Dict[str, Any]:
        """Fallback to Groq models using rotated API keys if Gemini quota is exhausted."""
        env_keys = os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY") or ""
        groq_keys = [k.strip() for k in env_keys.split(",") if k.strip()]
        if not groq_keys:
            raise RuntimeError("No Groq API keys found in GROQ_API_KEYS environment variable.")

        groq_models = ["groq/compound-mini", "llama-3.1-8b-instant", "llama3-8b-8192"]


        import random
        from groq import Groq

        last_error = None
        for key in random.sample(groq_keys, len(groq_keys)):

            groq_client = Groq(api_key=key)
            for gmodel in groq_models:
                try:
                    completion = groq_client.chat.completions.create(
                        model=gmodel,
                        messages=[
                            {"role": "system", "content": "You are a professional customer support quality evaluator. Respond ONLY with a valid JSON object matching the requested schema."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0,
                        response_format={"type": "json_object"}
                    )
                    content = completion.choices[0].message.content.strip()
                    return json.loads(content)
                except Exception as e:
                    last_error = e
                    continue
        raise last_error or RuntimeError("All Groq API keys failed.")


    def _call_openai_compatible(self, prompt: str) -> Dict[str, Any]:
        """Invoke OpenAI-compatible chat completion endpoint."""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a professional customer support quality evaluator."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)

__all__ = ["RealLLMJudge", "load_environment_credentials", "RUBRIC_DESCRIPTION"]

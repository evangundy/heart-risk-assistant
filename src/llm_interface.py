"""Natural-language layer: LLM parsing -> validation -> model -> LLM explanation.

The flow for every user message:

    1. EXTRACT   The LLM turns free text into JSON: intent + any feature values
                 it can see + anything ambiguous or unsupported.
    2. NORMALIZE Deterministic Python maps the LLM's wording to the exact codes
                 the model was trained on ("asymptomatic" -> "ASY", glucose
                 135 mg/dL -> FastingBS=1, cholesterol in mmol/L -> mg/dL ...).
    3. VALIDATE  Missing, ambiguous, or implausible values stop the flow and the
                 assistant asks a targeted follow-up question. The model is
                 never run on incomplete or guessed inputs.
    4. PREDICT   The trained model from MLflow produces a probability plus
                 per-feature impacts.
    5. EXPLAIN   The LLM writes the answer from the structured result only.
                 If the LLM is unavailable, a template answer is used.

Values persist across turns, so users can supply details over several
messages and ask "what if" follow-ups ("what if his max HR were 160?").
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.predictor import HeartRiskPredictor, Prediction

# --------------------------------------------------------------------------
# Feature specification: what the model needs, in words a user understands
# --------------------------------------------------------------------------
FEATURE_SPEC: dict[str, dict[str, Any]] = {
    "Age": {"label": "age", "ask": "the patient's age in years", "type": "num", "range": (18, 100)},
    "Sex": {"label": "sex", "ask": "sex (male or female)", "type": "cat", "levels": ["M", "F"]},
    "ChestPainType": {
        "label": "chest pain type", "type": "cat", "levels": ["TA", "ATA", "NAP", "ASY"],
        "ask": "chest pain type: typical angina, atypical angina, non-anginal pain, or none (asymptomatic)",
    },
    "RestingBP": {"label": "resting blood pressure", "ask": "resting systolic blood pressure (mm Hg)",
                  "type": "num", "range": (70, 220)},
    "Cholesterol": {"label": "cholesterol", "ask": "serum cholesterol (mg/dL), or say it wasn't measured",
                    "type": "num", "range": (85, 650), "nullable": True},
    "FastingBS": {"label": "fasting blood sugar", "type": "bin",
                  "ask": "fasting blood sugar: the value in mg/dL, or whether it is above 120 mg/dL"},
    "RestingECG": {"label": "resting ECG", "type": "cat", "levels": ["Normal", "ST", "LVH"],
                   "ask": "resting ECG result: normal, ST-T wave abnormality, or left ventricular hypertrophy"},
    "MaxHR": {"label": "max heart rate", "ask": "maximum heart rate reached during exercise testing (bpm)",
              "type": "num", "range": (60, 220)},
    "ExerciseAngina": {"label": "exercise-induced angina", "type": "cat", "levels": ["Y", "N"],
                       "ask": "whether exercise brought on chest pain (exercise-induced angina): yes or no"},
    "Oldpeak": {"label": "ST depression (oldpeak)", "type": "num", "range": (-3.0, 7.0),
                "ask": "ST depression during exercise relative to rest (oldpeak, in mm; 0 if none)"},
    "ST_Slope": {"label": "ST slope", "type": "cat", "levels": ["Up", "Flat", "Down"],
                 "ask": "slope of the peak exercise ST segment: upsloping, flat, or downsloping"},
}
REQUIRED = list(FEATURE_SPEC)

SYNONYMS: dict[str, dict[str, str]] = {
    "Sex": {"m": "M", "male": "M", "man": "M", "f": "F", "female": "F", "woman": "F"},
    "ChestPainType": {
        "ta": "TA", "typical": "TA", "typical angina": "TA",
        "ata": "ATA", "atypical": "ATA", "atypical angina": "ATA",
        "nap": "NAP", "non-anginal": "NAP", "non anginal": "NAP", "non-anginal pain": "NAP", "nonanginal": "NAP",
        "asy": "ASY", "asymptomatic": "ASY", "none": "ASY", "no chest pain": "ASY", "no pain": "ASY",
    },
    "RestingECG": {
        "normal": "Normal", "st": "ST", "st-t": "ST", "st-t wave abnormality": "ST", "st abnormality": "ST",
        "lvh": "LVH", "left ventricular hypertrophy": "LVH",
    },
    "ExerciseAngina": {"y": "Y", "yes": "Y", "true": "Y", "1": "Y", "n": "N", "no": "N", "false": "N", "0": "N"},
    "ST_Slope": {
        "up": "Up", "upsloping": "Up", "upward": "Up", "flat": "Flat", "down": "Down",
        "downsloping": "Down", "downward": "Down",
    },
}
NOT_MEASURED = {"not measured", "unknown",
                "not recorded", "n/a", "na", "missing", "none"}

# --------------------------------------------------------------------------
# LLM client
# --------------------------------------------------------------------------
DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"


class ChatLLM(Protocol):
    def chat(self, messages: list[dict], json_mode: bool = False,
             temperature: float = 0.0) -> str: ...


class NebiusLLM:
    """Thin wrapper around Nebius AI Studio's OpenAI-compatible API.

    Configuration comes only from environment variables (see .env.example):
    NEBIUS_API_KEY (required), NEBIUS_BASE_URL and NEBIUS_MODEL (optional).
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        from openai import OpenAI

        api_key = api_key or os.getenv("NEBIUS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NEBIUS_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.model = model or os.getenv("NEBIUS_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(api_key=api_key, base_url=base_url or os.getenv("NEBIUS_BASE_URL", DEFAULT_BASE_URL),
                             timeout=60)

    def chat(self, messages: list[dict], json_mode: bool = False, temperature: float = 0.0) -> str:
        kwargs = {"response_format": {
            "type": "json_object"}} if json_mode else {}
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=700, **kwargs
        )
        return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------
# Step 1: extraction
# --------------------------------------------------------------------------
EXTRACTION_PROMPT = """You are the input parser for a heart-disease risk model. Read the user's latest
message (plus the short conversation context) and return ONLY a JSON object:

{
  "intent": "predict" | "provide_info" | "model_question" | "reset" | "out_of_scope",
  "features": { <only features the user explicitly stated in THIS message> },
  "ambiguous": [ {"feature": "<name>", "reason": "<why it is unclear>"} ],
  "unsupported": [ "<facts the user gave that are not model features, e.g. smoking>" ]
}

Model features (use these exact keys):
- Age: years (number)
- Sex: "M" or "F"
- ChestPainType: "TA" typical angina, "ATA" atypical angina, "NAP" non-anginal pain, "ASY" asymptomatic/no chest pain
- RestingBP: resting systolic blood pressure in mm Hg (for "140/90" use 140)
- Cholesterol: serum cholesterol number as stated (keep the user's units; write "not measured" if they say so)
- FastingBS: fasting blood sugar. Give the number in mg/dL if stated, else true if above 120 mg/dL / diabetic-range, false if normal
- RestingECG: "Normal", "ST" (ST-T wave abnormality), or "LVH" (left ventricular hypertrophy)
- MaxHR: maximum heart rate achieved on exercise test (bpm)
- ExerciseAngina: "Y" or "N" (chest pain brought on by exercise)
- Oldpeak: ST depression in mm (number, can be 0)
- ST_Slope: "Up", "Flat", or "Down"

Rules:
- Never guess or fill in typical values. If a value is vague ("in his 50s", "high cholesterol",
  "BP is fine"), leave it out of "features" and add it to "ambiguous".
- If the user says "I"/"my", they are describing themselves; that is fine.
- "predict": they want a risk estimate (or gave patient details). "provide_info": they are answering a
  follow-up question. "model_question": they ask about this tool, its inputs, or its accuracy.
  "reset": they want to start over / a new patient. "out_of_scope": anything unrelated to estimating
  heart disease risk with this model (general medical advice, treatment, other diseases, chit-chat).
- A "what if" question that changes a value is "predict" with only the changed feature.
Return JSON only, no commentary."""


def parse_json(text: str) -> dict:
    """Parse JSON from an LLM reply, tolerating code fences or surrounding prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            return json.loads(brace.group(0))
        raise


# --------------------------------------------------------------------------
# Step 2: normalization (deterministic, unit-tested)
# --------------------------------------------------------------------------
def _to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        return float(m.group(0)) if m else None
    return None


def normalize_value(feature: str, value: Any) -> tuple[Any, str | None]:
    """Map one raw value to the model's encoding. Returns (value, error)."""
    spec = FEATURE_SPEC[feature]
    text = str(value).strip().lower() if value is not None else ""

    if feature == "Cholesterol" and (value is None or text in NOT_MEASURED):
        return None, None  # legitimately unmeasured -> imputed by the pipeline

    if feature == "FastingBS":
        if isinstance(value, bool):
            return int(value), None
        if text in {"yes", "true", "high", "elevated", "y", "diabetic", "above 120", ">120"}:
            return 1, None
        if text in {"no", "false", "normal", "n", "below 120", "<120"}:
            return 0, None
        num = _to_number(value)
        if num is None:
            return None, "fasting blood sugar should be a value in mg/dL or yes/no for above 120"
        if num in (0, 1):
            return int(num), None
        if num < 30:  # mmol/L -> mg/dL
            num *= 18.0
        return int(num > 120), None

    if spec["type"] == "cat":
        if isinstance(value, str) and value.strip() in spec["levels"]:
            return value.strip(), None
        mapped = SYNONYMS.get(feature, {}).get(text)
        if mapped:
            return mapped, None
        options = ", ".join(spec["levels"])
        return None, f"'{value}' is not a recognised {spec['label']} (expected one of: {options})"

    if feature == "Oldpeak" and text in {"none", "no", "no st depression", "no depression"}:
        return 0.0, None
    num = _to_number(value)
    if num is None:
        return None, f"could not read a number for {spec['label']} from '{value}'"
    if feature == "Cholesterol" and 1 < num < 20:  # mmol/L -> mg/dL
        num = round(num * 38.67)
    lo, hi = spec["range"]
    if not lo <= num <= hi:
        return None, f"{spec['label']} of {num:g} is outside the plausible range ({lo}-{hi})"
    return (int(num) if feature in {"Age", "RestingBP", "MaxHR", "Cholesterol"} else round(num, 1)), None


def normalize_features(raw: dict) -> tuple[dict, dict]:
    """Normalize every recognised key. Returns (clean_values, errors_by_feature)."""
    clean, errors = {}, {}
    for key, value in (raw or {}).items():
        feature = next((f for f in FEATURE_SPEC if f.lower()
                       == str(key).lower()), None)
        if feature is None:
            continue
        val, err = normalize_value(feature, value)
        if err:
            errors[feature] = err
        else:
            clean[feature] = val
    return clean, errors


def missing_features(known: dict) -> list[str]:
    return [f for f in REQUIRED if f not in known]


# --------------------------------------------------------------------------
# Step 5: response generation
# --------------------------------------------------------------------------
RESPONSE_PROMPT = """You explain the output of a heart-disease risk model to a clinician or an
informed patient. You receive JSON with the model's probability, risk band, the patient's inputs,
and the features that moved this prediction most (impact = change in probability versus a typical
patient in the training data; positive raises risk).

Write a clear answer of about 120-180 words in plain language:
1. Lead with the estimated probability as a percentage and the risk band.
2. Explain the 2-4 biggest drivers in everyday terms, saying which way each pushed the estimate.
   Briefly explain technical terms (e.g. ST slope, oldpeak) the first time you use them.
3. One sentence of context on what this means and a sensible next step (discuss with a clinician).
4. End with a one-line caveat: this is a statistical estimate from a model trained on 918 patients
   from historical hospital cohorts, not a diagnosis.
Use ONLY numbers present in the JSON. Do not invent factors, treatments, or statistics.
If unsupported facts were mentioned (e.g. smoking), note briefly that the model does not use them."""

MODEL_INFO_PROMPT = """You answer questions about a specific heart-disease risk tool. Facts you may use:
- It is a {model_type} classifier selected from {n_runs}+ tracked MLflow experiments.
- Test-set performance: {metrics}.
- Training data: Kaggle Heart Failure Prediction dataset, 918 patients combined from five historical
  hospital cohorts (Cleveland, Hungary, Switzerland, Long Beach VA, Statlog).
- Inputs: {features}.
- It estimates the probability of heart disease; it is not a diagnosis and gives no treatment advice.
Answer briefly (under 120 words) and stay within these facts. If asked something else, say it is
outside what this tool covers."""

SCOPE_MESSAGE = (
    "I can only help with one thing: estimating heart-disease risk from a patient's clinical "
    "measurements using a trained model. I can't give general medical advice, treatment "
    "recommendations, or answer unrelated questions.\n\n"
    "To get an estimate, describe the patient, for example: *\"58-year-old man, asymptomatic, "
    "BP 140, cholesterol 289, fasting sugar normal, normal ECG, max HR 125, exercise angina yes, "
    "oldpeak 1.5, flat ST slope.\"*"
)


def template_response(pred: Prediction, unsupported: list[str] | None = None) -> str:
    """LLM-free fallback so the app still answers if the API is down."""
    pct = round(pred.probability * 100)
    lines = [
        f"**Estimated probability of heart disease: {pct}% ({pred.risk_band} risk).**", ""]
    if pred.contributions:
        lines.append(
            "Biggest factors compared with a typical patient in the training data:")
        for c in pred.contributions:
            direction = "raised" if c["impact"] > 0 else "lowered"
            lines.append(f"- {FEATURE_SPEC[c['feature']]['label']} ({c['patient_value']}) "
                         f"{direction} the estimate by about {abs(c['impact']) * 100:.0f} points")
    if unsupported:
        lines.append(
            f"\nNote: the model does not use {', '.join(unsupported)}.")
    lines.append("\n_This is a statistical estimate from a model trained on 918 historical patients, "
                 "not a diagnosis. Please discuss results with a clinician._")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
@dataclass
class Reply:
    text: str
    kind: str  # prediction | clarify | invalid | out_of_scope | model_info | reset | error
    prediction: Prediction | None = None
    extracted: dict = field(default_factory=dict)


class HeartRiskAssistant:
    def __init__(self, predictor: HeartRiskPredictor, llm: ChatLLM | None):
        self.predictor = predictor
        self.llm = llm
        self.known: dict = {}
        self.history: list[dict] = []
        self.last_prediction: Prediction | None = None

    # -- public API ---------------------------------------------------------
    def reset(self) -> None:
        self.known, self.history, self.last_prediction = {}, [], None

    def set_features(self, values: dict) -> dict:
        """Set values directly (manual form); returns normalization errors."""
        clean, errors = normalize_features(values)
        self.known.update(clean)
        return errors

    def handle(self, message: str) -> Reply:
        self.history.append({"role": "user", "content": message})
        try:
            extraction = self.extract(message)
        except Exception as exc:  # network/JSON errors -> tell the user, don't crash
            return self._say(Reply(
                "Sorry, I couldn't reach the language model to read that message "
                f"({type(exc).__name__}). You can retry, or use the manual entry form.", "error"))

        intent = extraction.get("intent", "predict")
        raw_features = extraction.get("features") or {}
        ambiguous = extraction.get("ambiguous") or []
        unsupported = extraction.get("unsupported") or []

        if intent == "reset":
            self.reset()
            return self._say(Reply("Starting fresh. Tell me about the next patient.", "reset", extracted=extraction))
        if intent == "model_question":
            return self._say(Reply(self.answer_model_question(message), "model_info", extracted=extraction))
        if intent == "out_of_scope" and not raw_features:
            return self._say(Reply(SCOPE_MESSAGE, "out_of_scope", extracted=extraction))

        clean, errors = normalize_features(raw_features)
        self.known.update(clean)

        if errors:
            issues = "\n".join(f"- {msg}" for msg in errors.values())
            return self._say(Reply(
                f"I want to double-check a value before running the model:\n{issues}\n\n"
                "Could you confirm or correct it?", "invalid", extracted=extraction))

        ambiguous = [a for a in ambiguous if a.get(
            "feature") in FEATURE_SPEC and a["feature"] not in self.known]
        missing = missing_features(self.known)
        if ambiguous or missing:
            return self._say(Reply(self.clarification(missing, ambiguous, unsupported), "clarify",
                                   extracted=extraction))

        pred = self.predictor.predict(self.known)
        self.last_prediction = pred
        return self._say(Reply(self.explain(pred, unsupported), "prediction", prediction=pred, extracted=extraction))

    # -- steps ----------------------------------------------------------------
    def extract(self, message: str) -> dict:
        if self.llm is None:
            raise RuntimeError("No LLM configured")
        # brief context, excluding the current message
        context = [m for m in self.history[-6:-1]]
        known_note = f"Values already collected: {json.dumps(self.known)}" if self.known else "No values collected yet."
        messages = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "system", "content": known_note},
            *context,
            {"role": "user", "content": message},
        ]
        return parse_json(self.llm.chat(messages, json_mode=True))

    def clarification(self, missing: list[str], ambiguous: list[dict], unsupported: list[str]) -> str:
        parts = []
        if self.known:
            got = ", ".join(f"{FEATURE_SPEC[f]['label']} {v if v is not None else 'not measured'}"
                            for f, v in self.known.items())
            parts.append(f"Thanks. So far I have: {got}.")
        if ambiguous:
            parts.append("I need an exact value for: " + "; ".join(
                f"{FEATURE_SPEC[a['feature']]['label']} ({a.get('reason', 'unclear')})" for a in ambiguous) + ".")
        still = [f for f in missing if f not in {
            a['feature'] for a in ambiguous}]
        if still:
            parts.append("To run the model I still need:\n" +
                         "\n".join(f"- {FEATURE_SPEC[f]['ask']}" for f in still))
        if unsupported:
            parts.append(
                f"(Note: the model doesn't use {', '.join(unsupported)}.)")
        parts.append("I won't estimate risk until these are filled in, because guessed inputs would make "
                     "the result unreliable.")
        return "\n\n".join(parts)

    def explain(self, pred: Prediction, unsupported: list[str]) -> str:
        if self.llm is None:
            return template_response(pred, unsupported)
        payload = {**pred.to_dict(), "unsupported_facts_mentioned": unsupported}
        try:
            return self.llm.chat([
                {"role": "system", "content": RESPONSE_PROMPT},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ], temperature=0.3)
        except Exception:
            return template_response(pred, unsupported)

    def answer_model_question(self, question: str) -> str:
        meta = self.predictor.metadata
        system = MODEL_INFO_PROMPT.format(
            model_type=meta.get("model_type", "machine learning"),
            n_runs=5,
            metrics=", ".join(
                f"{k.replace('test_', '')} {v:.2f}" for k, v in meta["metrics"].items()),
            features=", ".join(FEATURE_SPEC[f]["label"] for f in REQUIRED),
        )
        try:
            return self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": question}],
                                 temperature=0.2)
        except Exception:
            return ("This tool estimates heart-disease probability from 11 clinical measurements using a "
                    f"{meta.get('model_type', 'machine learning')} model. It is not a diagnosis.")

    def _say(self, reply: Reply) -> Reply:
        self.history.append({"role": "assistant", "content": reply.text})
        return reply

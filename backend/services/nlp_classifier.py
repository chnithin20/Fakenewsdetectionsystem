import time
import re
import logging
import os
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_MODEL_PATH = str(BASE_DIR / "my-fakenews-model")

@dataclass
class NLPResult:
    label: str          # "real", "fake", or "suspicious"
    credibility_score: int
    confidence: float
    fake_probability: float
    real_probability: float
    sensationalism: float
    clickbait_probability: float
    emotional_language_index: float

# ── Sensationalism / Clickbait lexicon ────────────────────────────────────

SENSATIONALISM_WORDS = {
    "shocking", "outrageous", "unbelievable", "explosive", "bombshell",
    "exposed", "scandal", "secret", "hidden", "leaked", "breaking",
    "urgent", "must-see", "they don't want you to know", "wake up",
    "they're hiding", "mainstream media", "cover-up", "conspiracy",
    "miracle", "cure", "banned", "censored", "deep state",
    "you won't believe", "will blow your mind",
}

EMOTIONAL_AMPLIFIERS = {
    "never", "always", "every", "all", "none", "everyone", "nobody",
    "absolutely", "completely", "totally", "definitely", "obviously",
    "clearly", "undeniably", "catastrophic", "disastrous", "terrifying",
    "horrifying", "disgusting", "outrage", "fury", "rage", "crisis",
    "emergency", "danger", "threat", "destroy", "collapse",
}

CREDIBILITY_SIGNALS = {
    "according to", "study shows", "research indicates", "data suggests",
    "peer-reviewed", "published in", "official report", "spokesperson said",
    "confirmed by", "verified by", "sources say", "documents show",
    "statistics show", "percent", "survey found",
}

CLICKBAIT_PATTERNS = [
    r"\b(you won't believe|will shock you|mind\s*blown|jaw[\s-]?drop)\b",
    r"\b(number \d+ will)\b",
    r"\b(doctors hate|scientists hate|they don't want)\b",
    r"(\?{2,}|!{2,})",                  # Multiple ? or !
    r"\b(top \d+|this one trick|simple trick)\b",
    r"[A-Z]{4,}",                         # SHOUTING (4+ consecutive caps)
]

@lru_cache(maxsize=1)
def _get_classifier():
    """
    Load the transformers pipeline once and cache it.
    Prioritizes local fine-tuned model path.
    """
    model_path = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else settings.hf_model_name
    logger.info(f"Loading model from: {model_path}")
    
    try:
        from transformers import pipeline

        device = -1
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except Exception as torch_err:
            logger.warning(f"PyTorch unavailable, using CPU fallback: {torch_err}")

        return pipeline(
            "text-classification",
            model=model_path,
            device=device,
            model_kwargs={"cache_dir": settings.hf_model_cache_dir} if not os.path.exists(LOCAL_MODEL_PATH) else {}
        )
    except Exception as e:
        logger.error(f"Critical error loading model: {e}")
        return None


# ── Private helpers ────────────────────────────────────────────────────────

def _sensationalism_score(text: str) -> float:
    """Fraction of words that are sensationalism keywords (0.0 – 1.0)."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in SENSATIONALISM_WORDS)
    # Scale: 1 hit per 200 words ≈ 0.5 score. Cap at 1.0.
    return min(hits / max(len(words) / 200, 1), 1.0)


def _clickbait_score(text: str) -> float:
    """
    Returns a 0.0–1.0 score based on how many clickbait patterns are present.
    """
    score = 0.0
    for pattern in CLICKBAIT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            score += 0.2
    return min(score, 1.0)


def _emotional_language_score(text: str) -> float:
    """Fraction of sentences that contain emotional amplifiers."""
    sentences = re.split(r"[.!?]", text)
    if not sentences:
        return 0.0
    emotional = 0
    for sent in sentences:
        sent_lower = sent.lower()
        if any(amp in sent_lower for amp in EMOTIONAL_AMPLIFIERS):
            emotional += 1
    return round(emotional / len(sentences), 4)


def extract_signal_phrases(text: str) -> list[dict]:
    """
    Scan text for known signal words/phrases and return them with risk level.
    red   → strong fake indicator
    orange → moderate / ambiguous signal
    green  → credibility indicator
    """
    text_lower = text.lower()
    found: list[dict] = []
    seen: set[str] = set()

    for word in SENSATIONALISM_WORDS:
        if word in text_lower and word not in seen:
            found.append({"text": word, "level": "red"})
            seen.add(word)

    for pattern in CLICKBAIT_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for m in matches:
            phrase = m.strip().lower()
            if phrase and phrase not in seen:
                found.append({"text": phrase, "level": "red"})
                seen.add(phrase)

    for word in EMOTIONAL_AMPLIFIERS:
        if word in text_lower and word not in seen:
            found.append({"text": word, "level": "orange"})
            seen.add(word)

    for phrase in CREDIBILITY_SIGNALS:
        if phrase in text_lower and phrase not in seen:
            found.append({"text": phrase, "level": "green"})
            seen.add(phrase)

    # Cap at 12 signals for readability
    return found[:12]


def classify_text(text: str) -> NLPResult:
    """
    Run the full NLP pipeline using the fine-tuned local model.
    Logs inference time for each prediction.
    """
    start_time = time.time()
    try:
        classifier = _get_classifier()
        if classifier is None:
            raise ValueError("Model not loaded")

        # Truncate content for RoBERTa limit
        result = classifier(text[:1800])[0]
        
        # RoBERTa standard: LABEL_0 = REAL, LABEL_1 = FAKE
        # Note: hamzab/roberta uses id2label {"0": "FAKE", "1": "TRUE"}
        raw_label = result['label']
        confidence = result['score']
        logger.info(f"Raw model output: label={raw_label}, score={confidence}")
        
        # Normalize real/fake probabilities
        # RoBERTa standard from train.py: LABEL_0 = REAL, LABEL_1 = FAKE
        if raw_label.upper() in ["LABEL_0", "TRUE", "REAL", "TRUSTED"]:
            real_prob = confidence
            fake_prob = 1.0 - confidence
        elif raw_label.upper() in ["LABEL_1", "FAKE", "FALSE", "UNTRUSTED"]:
            fake_prob = confidence
            real_prob = 1.0 - confidence
        else:
            # Fallback if we don't recognize the label (e.g. from a different model)
            # Some models use '0' as FAKE and '1' as TRUE.
            # But after current fine-tuning, 0 is REAL and 1 is FAKE.
            logger.warning(f"Unrecognized model label: {raw_label}. Guessing based on common patterns.")
            if "0" in raw_label or "FAKE" in raw_label.upper():
                fake_prob = confidence
                real_prob = 1.0 - confidence
            else:
                real_prob = confidence
                fake_prob = 1.0 - confidence

        # ── 3-class threshold system ──
        credibility_score = int(real_prob * 100)
        
        if confidence < 0.55:
            label = "suspicious"
        elif credibility_score >= 65:
            label = "real"
        elif credibility_score >= 35:
            label = "suspicious"
        else:
            label = "fake"

        # Task 5.5: Log inference time
        inference_time = (time.time() - start_time) * 1000
        logger.info(f"Inference process: time={inference_time:.2f}ms | label={label} | score={credibility_score}")

    except Exception as e:
        logger.error(f"Inference failed ({e}). Using neutral fallback.")
        label = "suspicious"
        credibility_score = 50
        confidence = 0.5
        real_prob = 0.5
        fake_prob = 0.5

    # ── Step 2: Linguistic signals ──
    sens  = _sensationalism_score(text)
    click = _clickbait_score(text)
    emo   = _emotional_language_score(text)

    return NLPResult(
        label=label,
        credibility_score=credibility_score,
        confidence=round(confidence, 4),
        fake_probability=round(fake_prob, 4),
        real_probability=round(real_prob, 4),
        sensationalism=round(sens, 4),
        clickbait_probability=round(click, 4),
        emotional_language_index=round(emo, 4),
    )


# ── Task 1.5: Local Testing ───────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    headlines = [
        ("FAKE", "SHOCKING: Government CONFIRMS 5G towers injecting nanobots — They Don't Want You To Know!"),
        ("REAL", "The Federal Reserve on Wednesday held its key interest rate steady at a 23-year high for a sixth consecutive meeting, noting a 'lack of further progress' on inflation in recent months."),
        ("MIXED", "Anonymous Sources Say President Planning Major Policy Reversal Next Week as approval ratings dip.")
    ]
    
    # Force load model
    _get_classifier()
    
    print("\n" + "="*60)
    print("TASK 1: CLASSIFIER TEST RESULTS")
    print("="*60)
    
    for expected, text in headlines:
        res = classify_text(text)
        print(f"\nExpected: {expected}")
        print(f"Content:  {text[:80]}...")
        print(f"Result:   {res.label.upper()} (Score: {res.credibility_score}/100, Conf: {res.confidence})")
        print(f"Stats:    Fake={res.fake_probability}, Real={res.real_probability}")
        print("-" * 40)

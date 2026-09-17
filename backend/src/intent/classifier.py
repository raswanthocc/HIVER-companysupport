"""
Calibrated Multi-Class Intent Classifier and Baselines for CompanySupport.
Models are trained and calibrated EXCLUSIVELY on isolated non-Golden training data.
"""

import json
import os
import re
import time
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.pipeline import FeatureUnion

import google.generativeai as genai
from dotenv import load_dotenv

from src.models import IntentPrediction
from src.intent.taxonomy import INTENT_LABELS

def clean_text(text: str) -> str:
    """Normalize tweet text for classification."""
    text = text.lower()
    text = text.replace("i️", "i").replace("\ufe0f", "")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class TrivialClassifier:
    """Majority class baseline classifier."""
    def __init__(self, majority_class: str = "technical_issue"):
        self.majority_class = majority_class
        self.classes_ = INTENT_LABELS

    def fit(self, texts: List[str], labels: List[str]):
        counts = Counter(labels)
        if counts:
            self.majority_class = counts.most_common(1)[0][0]

    def predict(self, text: str) -> IntentPrediction:
        probs = {lbl: (1.0 if lbl == self.majority_class else 0.0) for lbl in INTENT_LABELS}
        return IntentPrediction(intent=self.majority_class, confidence=1.0, probabilities=probs)

    def predict_batch(self, texts: List[str]) -> List[IntentPrediction]:
        return [self.predict(t) for t in texts]


class SimpleClassifier:
    """Simple baseline classifier: Word TF-IDF + uncalibrated Logistic Regression."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 1))
        self.model = LogisticRegression(max_iter=500, random_state=42)
        self.classes_ = INTENT_LABELS
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        cleaned = [clean_text(t) for t in texts]
        X = self.vectorizer.fit_transform(cleaned)
        y_arr = np.array(labels)
        self.model.fit(X, y_arr)
        self.classes_ = list(self.model.classes_)
        self.is_fitted = True

    def predict(self, text: str) -> IntentPrediction:
        if not self.is_fitted:
            raise RuntimeError("SimpleClassifier is not fitted. Call fit() with training data first.")
        cleaned = clean_text(text)
        X = self.vectorizer.transform([cleaned])
        probs_raw = self.model.predict_proba(X)[0]
        probs = {c: float(p) for c, p in zip(self.classes_, probs_raw)}
        best_intent = self.model.predict(X)[0]
        confidence = float(np.max(probs_raw))
        return IntentPrediction(intent=best_intent, confidence=confidence, probabilities=probs)

    def predict_batch(self, texts: List[str]) -> List[IntentPrediction]:
        return [self.predict(t) for t in texts]


class CalibratedIntentClassifier:
    """
    Proposed System Intent Classifier:
    - FeatureUnion of word n-grams (1, 2) and character n-grams (3, 5) with sublinear TF-IDF
    - Balanced class weights to handle natural class skew
    - Probability calibration via CalibratedClassifierCV
    - STRICTLY fits only on training split
    """
    def __init__(self):
        self.vectorizer = FeatureUnion([
            ("word_ngram", TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=8000,
                stop_words="english",
                token_pattern=r"(?u)\b\w+\b"
            )),
            ("char_ngram", TfidfVectorizer(
                ngram_range=(3, 5),
                analyzer="char_wb",
                sublinear_tf=True,
                max_features=8000
            ))
        ])

        self.base_lr = LogisticRegression(
            C=3.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        )
        self.models = {}  # Dict[str, CalibratedClassifierCV | LogisticRegression]
        self.global_model = None # Fallback if brand unseen
        self.classes_ = INTENT_LABELS
        self.is_fitted = False

        # Initialize Gemini for fallback
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "..", ".env")
        load_dotenv(env_path)
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))

    def fit(self, texts: List[str], labels: List[str], brands: List[str] = None):
        if not brands:
            brands = ["Unknown"] * len(texts)

        cleaned = [clean_text(t) for t in texts]
        X = self.vectorizer.fit_transform(cleaned)
        y_arr = np.array(labels)
        
        # Train global fallback model
        counts = Counter(labels)
        min_count = min(counts.values()) if counts else 0
        if min_count >= 3 and len(labels) >= 18:
            self.global_model = CalibratedClassifierCV(estimator=self.base_lr, cv=3)
            self.global_model.fit(X, y_arr)
        else:
            self.global_model = self.base_lr
            self.global_model.fit(X, y_arr)

        # Train brand-specific models
        brand_indices = defaultdict(list)
        for i, b in enumerate(brands):
            brand_indices[b].append(i)

        for b, indices in brand_indices.items():
            X_b = X[indices]
            y_b = y_arr[indices]
            counts_b = Counter(y_b)
            min_count_b = min(counts_b.values()) if counts_b else 0
            
            # Need multiple classes to train LR
            if len(counts_b) > 1:
                from sklearn.base import clone
                model_b = clone(self.base_lr)
                if min_count_b >= 3 and len(indices) >= 18:
                    calibrated = CalibratedClassifierCV(estimator=model_b, cv=3)
                    calibrated.fit(X_b, y_b)
                    self.models[b] = calibrated
                else:
                    model_b.fit(X_b, y_b)
                    self.models[b] = model_b

        self.classes_ = list(self.global_model.classes_)
        self.is_fitted = True

    def auto_fit_if_needed(self):
        """Auto-fit using ONLY data/train_data.json if not already fitted. Never touches Golden data."""
        if self.is_fitted:
            return
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        train_path = os.path.join(base_dir, "data", "train_data.json")
        if os.path.exists(train_path):
            with open(train_path, encoding="utf-8") as f:
                data = json.load(f)
            texts = [d["customer_text"] for d in data]
            labels = [d["intent"] for d in data]
            brands = [d.get("brand", "Unknown") for d in data]
            self.fit(texts, labels, brands)
        else:
            raise RuntimeError(f"Training data not found at {train_path}. Run data/split.py first.")

    def predict(self, text: str, brand: str = "Unknown") -> IntentPrediction:
        if not self.is_fitted:
            self.auto_fit_if_needed()

        cleaned = clean_text(text)
        X = self.vectorizer.transform([cleaned])

        active_model = self.models.get(brand, self.global_model)
        
        probs_raw = active_model.predict_proba(X)[0]
        model_classes = list(active_model.classes_)
        probs = {c: float(p) for c, p in zip(model_classes, probs_raw)}
        
        # Fill missing classes with 0.0 for consistent schema
        for c in INTENT_LABELS:
            if c not in probs:
                probs[c] = 0.0

        best_intent = model_classes[int(np.argmax(probs_raw))]
        confidence = float(np.max(probs_raw))
        
        # Confidence-Gated Hybrid Routing (LLM Fallback)
        if confidence < 0.75 and self.llm:
            try:
                prompt = (
                    f"Classify the following customer query into exactly ONE of these intents:\n"
                    f"{', '.join(INTENT_LABELS)}\n\n"
                    f"Query: '{text}'\n"
                    f"Respond ONLY with the exact intent string from the list above. No other text."
                )
                response = self.llm.generate_content(prompt)
                llm_intent = response.text.strip()
                if llm_intent in INTENT_LABELS:
                    best_intent = llm_intent
                    confidence = 0.99  # Indicate it came from LLM but is high confidence
                    probs = {lbl: 0.99 if lbl == best_intent else 0.01/19.0 for lbl in INTENT_LABELS}
            except Exception as e:
                # Groq Fallback Logic
                env_keys = os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY") or ""
                groq_keys = [k.strip() for k in env_keys.split(",") if k.strip()]
                if groq_keys:
                    groq_models = ["groq/compound-mini", "llama-3.1-8b-instant", "llama3-8b-8192"]
                    from groq import Groq
                    import random


                groq_success = False
                for key in random.sample(groq_keys, len(groq_keys)):
                    groq_client = Groq(api_key=key)
                    for gmodel in groq_models:
                        try:
                            completion = groq_client.chat.completions.create(
                                model=gmodel,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.0,
                                max_tokens=32
                            )
                            llm_intent = completion.choices[0].message.content.strip()
                            if llm_intent in INTENT_LABELS:
                                best_intent = llm_intent
                                confidence = 0.99
                                probs = {lbl: 0.99 if lbl == best_intent else 0.01/19.0 for lbl in INTENT_LABELS}
                                groq_success = True
                            break
                        except Exception:
                            continue
                    if groq_success:
                        break



                if not groq_success:
                    pass # Safely fallback to Logistic Regression prediction

        return IntentPrediction(intent=best_intent, confidence=confidence, probabilities=probs)

    def predict_batch(self, texts: List[str], brands: List[str] = None) -> List[IntentPrediction]:
        if not brands:
            brands = ["Unknown"] * len(texts)
        return [self.predict(t, b) for t, b in zip(texts, brands)]

    def evaluate(self, texts: List[str], ground_truth: List[str], brands: List[str] = None) -> Dict[str, Any]:
        if not brands:
            brands = ["Unknown"] * len(texts)
        preds = [self.predict(t, b).intent for t, b in zip(texts, brands)]
        acc = float(accuracy_score(ground_truth, preds))
        macro_f1 = float(f1_score(ground_truth, preds, average="macro", zero_division=0))
        precision = float(precision_score(ground_truth, preds, average="macro", zero_division=0))
        recall = float(recall_score(ground_truth, preds, average="macro", zero_division=0))
        cm = confusion_matrix(ground_truth, preds, labels=INTENT_LABELS).tolist()
        report = classification_report(ground_truth, preds, labels=INTENT_LABELS, output_dict=True, zero_division=0)

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "macro_precision": precision,
            "macro_recall": recall,
            "confusion_matrix": cm,
            "per_class_report": report
        }

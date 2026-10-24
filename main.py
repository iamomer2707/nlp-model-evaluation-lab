from __future__ import annotations

import csv
import io
import json
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
DATA_FILE = ROOT / "data" / "benchmark_results.json"
UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
MAX_TEXT_LENGTH = 3_000
MAX_CSV_BYTES = 2_000_000

app = FastAPI(title="Signal Lab API", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_model_lock = threading.Lock()
_demo_model: Pipeline | None = None
_demo_classes: list[str] | None = None
_demo_metrics: dict | None = None
_demo_error: str | None = None


class PredictionRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)


class EvaluationRequest(BaseModel):
    csv_text: str = Field(min_length=1, max_length=MAX_CSV_BYTES)
    filename: str = Field(default="dataset.csv", max_length=120)


def make_classifier() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=100_000,
        )),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            random_state=42,
        )),
    ])


def read_uci_messages() -> tuple[list[str], list[str]]:
    request = urllib.request.Request(UCI_URL, headers={"User-Agent": "SignalLab/1.0"})
    with urllib.request.urlopen(request, timeout=25) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        member = next(name for name in archive.namelist()
                      if name.rsplit("/", 1)[-1] == "SMSSpamCollection")
        text = archive.read(member).decode("utf-8", errors="replace")

    messages, labels = [], []
    for line in text.splitlines():
        if "\t" not in line:
            continue
        label, message = line.split("\t", maxsplit=1)
        if label in {"ham", "spam"} and message.strip():
            labels.append(label)
            messages.append(message.strip())
    if len(messages) < 100:
        raise RuntimeError("The SMS Spam Collection download was incomplete.")
    return messages, labels


def score_model(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict:
    positive_label = classes[-1]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "recall": float(recall_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "f1": float(f1_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "positive_label": positive_label,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "class_order": classes,
    }


def initialize_demo() -> None:
    global _demo_model, _demo_classes, _demo_metrics, _demo_error
    if _demo_model is not None or _demo_error is not None:
        return
    with _model_lock:
        if _demo_model is not None or _demo_error is not None:
            return
        try:
            messages, labels = read_uci_messages()
            x_train, x_remaining, y_train, y_remaining = train_test_split(
                messages, labels, test_size=0.30, random_state=42, stratify=labels)
            x_validation, x_test, y_validation, y_test = train_test_split(
                x_remaining, y_remaining, test_size=0.50,
                random_state=42, stratify=y_remaining)
            model = make_classifier()
            model.fit(x_train, y_train)
            predictions = model.predict(x_test).tolist()
            classes = sorted(set(labels))
            metrics = score_model(y_test, predictions, classes)
            metrics.update({
                "messages": len(messages),
                "train_messages": len(x_train),
                "validation_messages": len(x_validation),
                "test_messages": len(x_test),
            })
            _demo_model, _demo_classes, _demo_metrics = model, classes, metrics
        except Exception:
            # The benchmark remains viewable if the user's host has no network.
            _demo_error = "Live classification needs internet access the first time it loads the public UCI dataset."


def benchmark_payload() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "online"}


@app.get("/api/overview")
def overview() -> dict:
    initialize_demo()
    return {
        "benchmark": benchmark_payload(),
        "demo_ready": _demo_model is not None,
        "demo_metrics": _demo_metrics,
        "demo_error": _demo_error,
        "demo_model": "TF-IDF + Logistic Regression",
    }


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict:
    initialize_demo()
    if _demo_model is None or _demo_classes is None:
        raise HTTPException(status_code=503, detail=_demo_error or "The demo classifier is unavailable.")

    started = time.perf_counter()
    predicted = str(_demo_model.predict([request.message])[0])
    probabilities = _demo_model.predict_proba([request.message])[0]
    class_probabilities = {
        str(label): float(probability)
        for label, probability in zip(_demo_model.classes_, probabilities)
    }
    positive_label = _demo_classes[-1]
    return {
        "prediction": predicted,
        "positive_label": positive_label,
        "positive_probability": class_probabilities.get(positive_label, 0.0),
        "confidence": class_probabilities.get(predicted, 0.0),
        "class_probabilities": class_probabilities,
        "latency_ms": (time.perf_counter() - started) * 1_000,
    }


def parse_binary_dataset(csv_text: str) -> tuple[list[str], list[str], list[str]]:
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
    if not reader.fieldnames:
        raise ValueError("The CSV needs a header row with text and label columns.")
    columns = {name.strip().lower(): name for name in reader.fieldnames if name}
    text_column = next((columns[key] for key in ("text", "message", "sms", "content") if key in columns), None)
    label_column = next((columns[key] for key in ("label", "target", "class", "category") if key in columns), None)
    if text_column is None or label_column is None:
        raise ValueError("Use a text/message column and a label/class column.")

    messages, labels = [], []
    for row in reader:
        message = (row.get(text_column) or "").strip()
        label = (row.get(label_column) or "").strip()
        if message and label:
            messages.append(message[:MAX_TEXT_LENGTH])
            labels.append(label)
    classes = sorted(set(labels))
    if len(classes) != 2:
        raise ValueError(f"Expected exactly two label values; found {len(classes)}.")
    if len(messages) < 20 or min(labels.count(value) for value in classes) < 5:
        raise ValueError("Use at least 20 non-empty rows and at least 5 examples per class.")
    return messages, labels, classes


@app.post("/api/evaluate")
def evaluate_csv(request: EvaluationRequest) -> dict:
    if len(request.csv_text.encode("utf-8")) > MAX_CSV_BYTES:
        raise HTTPException(status_code=413, detail="CSV is larger than the 2 MB limit.")
    try:
        messages, labels, classes = parse_binary_dataset(request.csv_text)
        x_train, x_test, y_train, y_test = train_test_split(
            messages, labels, test_size=0.25, random_state=42, stratify=labels)
        model = make_classifier()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test).tolist()
        metrics = score_model(y_test, predictions, classes)
        metrics.update({"rows": len(messages), "train_rows": len(x_train), "test_rows": len(x_test)})
        return metrics
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=422, detail="Could not evaluate this CSV. Check its encoding and columns.") from error


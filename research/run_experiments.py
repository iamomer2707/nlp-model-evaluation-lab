"""Train Assignment 2 baselines and four pretrained classifiers on SMS spam.

Run in Google Colab with a GPU runtime, or locally after installing the
dependencies listed in the notebook. Results and figures are written to
./outputs/.
"""
from __future__ import annotations

import copy
import json
import random
import re
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------- Configuration ----------------------------
SEED = 42
MAX_LENGTH = 128
SCRATCH_MAX_TOKENS = 100
SCRATCH_EPOCHS = 8
PRETRAINED_EPOCHS = 4
EARLY_STOPPING_PATIENCE = 2
SCRATCH_BATCH_SIZE = 128
PRETRAINED_BATCH_SIZE = 16
PRETRAINED_LEARNING_RATE = 2e-5
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_DIR = Path("work/data/uci")
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRETRAINED_MODELS = {
    "DistilBERT": "distilbert/distilbert-base-uncased",
    "BERT": "google-bert/bert-base-uncased",
    "RoBERTa": "FacebookAI/roberta-base",
    "MiniLM": "microsoft/MiniLM-L12-H384-uncased",
}

# Exact Assignment 2 held-out metrics supplied in experiment_results.json.
# Its split was 70/10/20, so these are historical reference values rather
# than results from the matched 70/15/15 rerun below.
ASSIGNMENT_2_RESULTS = {
    "RNN": {"accuracy": 0.9210762331838565, "precision_spam": 0.6380090497737556,
            "recall_spam": 0.9463087248322147, "f1_spam": 0.7621621621621621,
            "training_seconds": 1.480581040999823},
    "LSTM": {"accuracy": 0.9730941704035875, "precision_spam": 0.9219858156028369,
             "recall_spam": 0.87248322147651, "f1_spam": 0.896551724137931,
             "training_seconds": 2.715169541999785},
    "LSTM + Attention": {"accuracy": 0.95695067264574, "precision_spam": 0.7759562841530054,
                          "recall_spam": 0.9530201342281879, "f1_spam": 0.8554216867469879,
                          "training_seconds": 5.188917416999175},
    "Transformer": {"accuracy": 0.9533632286995516, "precision_spam": 0.793939393939394,
                    "recall_spam": 0.8791946308724832, "f1_spam": 0.8343949044585988,
                    "training_seconds": 17.500974000002316},
}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def load_sms_dataset() -> pd.DataFrame:
    """Read the same UCI SMS Spam Collection used in Assignment 2."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_file = DATA_DIR / "SMSSpamCollection"
    if not data_file.exists():
        zip_file = DATA_DIR / "sms-spam-collection.zip"
        urllib.request.urlretrieve(DATA_URL, zip_file)
        with zipfile.ZipFile(zip_file) as archive:
            member = next(name for name in archive.namelist()
                          if name.rsplit("/", 1)[-1] == "SMSSpamCollection")
            data_file.write_bytes(archive.read(member))

    rows = []
    with data_file.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            label_text, message = line.rstrip("\r\n").split("\t", maxsplit=1)
            rows.append((message, 1 if label_text == "spam" else 0))
    frame = pd.DataFrame(rows, columns=["message", "label"])
    frame["label"] = frame["label"].astype("int64")
    return frame


def split_data(frame: pd.DataFrame):
    """Create stratified 70/15/15 train, validation, and test partitions."""
    train, remainder = train_test_split(
        frame, test_size=0.30, random_state=SEED, stratify=frame["label"])
    validation, test = train_test_split(
        remainder, test_size=0.50, random_state=SEED, stratify=remainder["label"])
    return tuple(part.reset_index(drop=True) for part in (train, validation, test))


def score_predictions(y_true, y_pred) -> dict:
    """Use Evaluate for the requested metrics and sklearn for the matrix."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    try:
        accuracy = evaluate.load("accuracy").compute(
            predictions=y_pred, references=y_true)["accuracy"]
        precision = evaluate.load("precision").compute(
            predictions=y_pred, references=y_true, average="binary",
            pos_label=1, zero_division=0)["precision"]
        recall = evaluate.load("recall").compute(
            predictions=y_pred, references=y_true, average="binary",
            pos_label=1, zero_division=0)["recall"]
        f1 = evaluate.load("f1").compute(
            predictions=y_pred, references=y_true, average="binary",
            pos_label=1, zero_division=0)["f1"]
    except Exception as exc:
        # The fallback keeps the experiment usable if Evaluate cannot fetch
        # its small metric scripts in a restricted runtime.
        print(f"Evaluate metric loading failed ({exc}); using sklearn metrics.")
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy),
        "precision_spam": float(precision),
        "recall_spam": float(recall),
        "f1_spam": float(f1),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


# -------------------------- Assignment 2 models -------------------------
def clean_and_tokenize(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return text.split()


def build_vocabulary(train_frame: pd.DataFrame) -> dict[str, int]:
    counts = Counter(token for message in train_frame["message"]
                     for token in clean_and_tokenize(message))
    # Stable tie-breaking makes the vocabulary reproducible.
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {"<pad>": 0, "<unk>": 1,
            **{token: index + 2 for index, (token, _) in enumerate(ordered)}}


def vectorize_message(message: str, vocab: dict[str, int]):
    tokens = clean_and_tokenize(message)[:SCRATCH_MAX_TOKENS]
    ids = [vocab.get(token, 1) for token in tokens]
    if not ids:
        ids = [1]
    length = len(ids)
    ids += [0] * (SCRATCH_MAX_TOKENS - len(ids))
    return ids, length


class SMSVectorDataset(TorchDataset):
    def __init__(self, frame: pd.DataFrame, vocab: dict[str, int]):
        self.items = []
        for row in frame.itertuples(index=False):
            ids, length = vectorize_message(row.message, vocab)
            self.items.append((torch.tensor(ids, dtype=torch.long), length,
                               int(row.label)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class RecurrentClassifier(nn.Module):
    def __init__(self, vocab_size: int, kind: str):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        recurrent = nn.RNN if kind == "rnn" else nn.LSTM
        self.encoder = recurrent(64, 64, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(64, 2)

    def forward(self, tokens, lengths):
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, state = self.encoder(packed)
        hidden = state[0] if self.kind == "lstm" else state
        return self.classifier(self.dropout(hidden[-1]))


class LSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.encoder = nn.LSTM(64, 64, batch_first=True)
        self.attention = nn.Linear(64, 1, bias=False)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(64, 2)

    def forward(self, tokens, lengths):
        outputs, _ = self.encoder(self.embedding(tokens))
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        mask = positions < lengths.to(tokens.device).unsqueeze(1)
        scores = self.attention(outputs).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(outputs * weights.unsqueeze(-1), dim=1)
        return self.classifier(self.dropout(context))


class ScratchTransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.position = nn.Embedding(SCRATCH_MAX_TOKENS, 64)
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.15,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2,
                                             enable_nested_tensor=False)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(64, 2)

    def forward(self, tokens, lengths):
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        hidden = self.embedding(tokens) + self.position(positions)
        padding_mask = tokens.eq(0)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return self.classifier(self.dropout(pooled))


def build_scratch_model(name: str, vocab_size: int):
    if name == "RNN":
        return RecurrentClassifier(vocab_size, "rnn")
    if name == "LSTM":
        return RecurrentClassifier(vocab_size, "lstm")
    if name == "LSTM + Attention":
        return LSTMAttentionClassifier(vocab_size)
    if name == "Transformer":
        return ScratchTransformerClassifier(vocab_size)
    raise ValueError(name)


def run_scratch_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_all, predictions_all = [], []
    for tokens, lengths, labels in loader:
        tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(tokens, lengths)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss.item() * len(labels)
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(logits.detach().argmax(dim=1).cpu().tolist())
    average_loss = total_loss / len(loader.dataset)
    f1 = f1_score(labels_all, predictions_all, zero_division=0)
    return average_loss, f1, labels_all, predictions_all


def predict_scratch(model, loader):
    model.eval()
    labels_all, predictions_all, probabilities = [], [], []
    started = time.perf_counter()
    with torch.no_grad():
        for tokens, lengths, labels in loader:
            logits = model(tokens.to(DEVICE), lengths)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probabilities.extend(probs.tolist())
            predictions_all.extend((probs >= 0.5).astype(int).tolist())
            labels_all.extend(labels.numpy().tolist())
    elapsed = time.perf_counter() - started
    return np.asarray(labels_all), np.asarray(predictions_all), np.asarray(probabilities), elapsed


def train_scratch_models(train_frame, validation_frame, test_frame):
    seed_everything()
    vocab = build_vocabulary(train_frame)
    train_ds, validation_ds, test_ds = [
        SMSVectorDataset(part, vocab)
        for part in (train_frame, validation_frame, test_frame)]
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=SCRATCH_BATCH_SIZE,
                              shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_ds, batch_size=SCRATCH_BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=SCRATCH_BATCH_SIZE)
    counts = np.bincount(train_frame["label"].to_numpy(), minlength=2)
    class_weights = torch.tensor(len(train_frame) / (2 * counts),
                                 dtype=torch.float32, device=DEVICE)

    results, histories = {}, {}
    for name in ["RNN", "LSTM", "LSTM + Attention", "Transformer"]:
        seed_everything()
        model = build_scratch_model(name, len(vocab)).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = {"train_loss": [], "validation_loss": [], "validation_f1_spam": []}
        best_key, best_state, stale = (-float("inf"), float("inf")), None, 0
        start = time.perf_counter()
        for epoch in range(SCRATCH_EPOCHS):
            train_loss, _, _, _ = run_scratch_epoch(
                model, train_loader, criterion, optimizer)
            val_loss, val_f1, _, _ = run_scratch_epoch(
                model, validation_loader, criterion)
            history["train_loss"].append(float(train_loss))
            history["validation_loss"].append(float(val_loss))
            history["validation_f1_spam"].append(float(val_f1))
            key = (val_f1, -val_loss)
            print(f"{name} epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                  f"validation_loss={val_loss:.4f}, validation_spam_f1={val_f1:.4f}")
            if key > best_key:
                best_key = key
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= EARLY_STOPPING_PATIENCE:
                    break
        training_seconds = time.perf_counter() - start
        model.load_state_dict(best_state)
        y_true, y_pred, y_prob, inference_seconds = predict_scratch(model, test_loader)
        row = score_predictions(y_true, y_pred)
        row.update({
            "model_family": "from_scratch",
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "training_seconds": float(training_seconds),
            "inference_seconds": float(inference_seconds),
            "best_validation_f1_spam": float(best_key[0]),
            "epochs_trained": len(history["train_loss"]),
        })
        results[name] = row
        row["y_true"] = y_true.tolist()
        row["y_pred"] = y_pred.tolist()
        row["spam_probability"] = y_prob.tolist()
        histories[name] = history
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results, histories, vocab


# ------------------------- Pretrained classifiers -----------------------
def make_hf_loader(frame, tokenizer, shuffle=False):
    dataset = Dataset.from_pandas(
        frame[["message", "label"]].reset_index(drop=True), preserve_index=False)
    dataset = dataset.map(
        lambda batch: tokenizer(batch["message"], truncation=True,
                                 padding="max_length", max_length=MAX_LENGTH),
        batched=True, remove_columns=["message"])
    columns = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in dataset.column_names:
        columns.append("token_type_ids")
    dataset.set_format(type="torch", columns=columns)
    generator = torch.Generator().manual_seed(SEED) if shuffle else None
    return DataLoader(dataset, batch_size=PRETRAINED_BATCH_SIZE,
                      shuffle=shuffle, generator=generator)


def run_pretrained_validation(model, loader, class_weights):
    model.eval()
    total_loss = 0.0
    labels_all, predictions_all = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(DEVICE)
            inputs = {key: value.to(DEVICE) for key, value in batch.items()
                      if key != "label"}
            logits = model(**inputs).logits
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            total_loss += loss.item() * len(labels)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(logits.argmax(dim=1).cpu().tolist())
    return (total_loss / len(loader.dataset),
            float(f1_score(labels_all, predictions_all, zero_division=0)))


def predict_pretrained(model, loader):
    model.eval()
    labels_all, predictions_all, probabilities = [], [], []
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(DEVICE)
            inputs = {key: value.to(DEVICE) for key, value in batch.items()
                      if key != "label"}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probabilities.extend(probs.tolist())
            predictions_all.extend((probs >= 0.5).astype(int).tolist())
            labels_all.extend(labels.cpu().tolist())
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return np.asarray(labels_all), np.asarray(predictions_all), np.asarray(probabilities), elapsed


def train_pretrained_models(train_frame, validation_frame, test_frame):
    results, histories = {}, {}
    train_counts = np.bincount(train_frame["label"].to_numpy(), minlength=2)
    class_weights = torch.tensor(len(train_frame) / (2 * train_counts),
                                 dtype=torch.float32, device=DEVICE)

    for name, checkpoint in PRETRAINED_MODELS.items():
        seed_everything()
        print(f"\nLoading {name} from {checkpoint}")
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=2, id2label={0: "HAM", 1: "SPAM"},
            label2id={"HAM": 0, "SPAM": 1})
        model.to(DEVICE)
        model_load_seconds = time.perf_counter() - load_start
        train_loader = make_hf_loader(train_frame, tokenizer, shuffle=True)
        validation_loader = make_hf_loader(validation_frame, tokenizer)
        test_loader = make_hf_loader(test_frame, tokenizer)
        optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAINED_LEARNING_RATE,
                                      weight_decay=0.01)
        history = {"train_loss": [], "validation_loss": [], "validation_f1_spam": []}
        best_key, best_state, stale = (-float("inf"), float("inf")), None, 0
        start = time.perf_counter()
        for epoch in range(PRETRAINED_EPOCHS):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                labels = batch["label"].to(DEVICE)
                inputs = {key: value.to(DEVICE) for key, value in batch.items()
                          if key != "label"}
                optimizer.zero_grad(set_to_none=True)
                logits = model(**inputs).logits
                loss = F.cross_entropy(logits, labels, weight=class_weights)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(labels)
            train_loss = total_loss / len(train_loader.dataset)
            val_loss, val_f1 = run_pretrained_validation(
                model, validation_loader, class_weights)
            history["train_loss"].append(float(train_loss))
            history["validation_loss"].append(float(val_loss))
            history["validation_f1_spam"].append(float(val_f1))
            key = (val_f1, -val_loss)
            print(f"{name} epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                  f"validation_loss={val_loss:.4f}, validation_spam_f1={val_f1:.4f}")
            if key > best_key:
                best_key = key
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= EARLY_STOPPING_PATIENCE:
                    break
        training_seconds = time.perf_counter() - start
        model.load_state_dict(best_state)

        # The test partition is used only here, after model selection.
        y_true, y_pred, y_prob, inference_seconds = predict_pretrained(model, test_loader)
        row = score_predictions(y_true, y_pred)
        row.update({
            "model_family": "pretrained_finetuned",
            "checkpoint": checkpoint,
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "estimated_weights_mb_fp32": float(
                sum(parameter.numel() for parameter in model.parameters()) * 4 / (1024 ** 2)),
            "model_load_seconds": float(model_load_seconds),
            "training_seconds": float(training_seconds),
            "inference_seconds": float(inference_seconds),
            "best_validation_f1_spam": float(best_key[0]),
            "epochs_trained": len(history["train_loss"]),
        })
        row["y_true"] = y_true.tolist()
        row["y_pred"] = y_pred.tolist()
        row["spam_probability"] = y_prob.tolist()
        results[name] = row
        histories[name] = history
        del model, tokenizer, train_loader, validation_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results, histories


# --------------------------- Reports and plots --------------------------
def choose_error_examples(results):
    best_name = max(PRETRAINED_MODELS,
                    key=lambda name: results[name]["best_validation_f1_spam"])
    result = results[best_name]
    examples = pd.DataFrame({
        "message": test_frame_global["message"],
        "actual": result["y_true"],
        "predicted": result["y_pred"],
        "spam_probability": result["spam_probability"],
    })
    examples["case_type"] = np.select(
        [examples.actual.eq(0) & examples.predicted.eq(0),
         examples.actual.eq(1) & examples.predicted.eq(1),
         examples.actual.eq(0) & examples.predicted.eq(1)],
        ["correct_ham", "correct_spam", "false_positive"], default="false_negative")
    examples["confidence"] = np.where(
        examples.predicted.eq(1), examples.spam_probability,
        1 - examples.spam_probability)

    correct = examples[examples.actual.eq(examples.predicted)].copy()
    correct = correct.sort_values("confidence", ascending=False).head(5)
    errors = examples[examples.actual.ne(examples.predicted)].copy()
    # Prefer a balanced sample of false positives and false negatives, then
    # fill any shortage with the remaining highest-confidence errors.
    false_positive = errors[errors.case_type.eq("false_positive")].sort_values(
        "confidence", ascending=False).head(3)
    false_negative = errors[errors.case_type.eq("false_negative")].sort_values(
        "confidence", ascending=False).head(3)
    incorrect = pd.concat([false_positive, false_negative]).drop_duplicates()
    if len(incorrect) < 5:
        remaining = errors.drop(index=incorrect.index).sort_values(
            "confidence", ascending=False).head(5 - len(incorrect))
        incorrect = pd.concat([incorrect, remaining])
    correct.to_csv(OUTPUT_DIR / "five_correct_predictions.csv", index=False)
    incorrect.to_csv(OUTPUT_DIR / "five_incorrect_predictions.csv", index=False)
    return best_name, correct, incorrect


def plot_loss_curves(histories):
    names = list(histories)
    fig, axes = plt.subplots(4, 2, figsize=(12, 15))
    for ax, name in zip(axes.flat, names):
        history = histories[name]
        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], marker="o", label="Train")
        ax.plot(epochs, history["validation_loss"], marker="o", label="Validation")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted cross-entropy loss")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle("Training and validation loss", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(FIGURE_DIR / "loss_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(results):
    names = list(results)
    values = [results[name]["accuracy"] for name in names]
    colors = ["#7b8794"] * 4 + ["#167d9a"] * 4
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(names, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy on the shared stratified 15% test split", weight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "accuracy_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(results):
    names = list(results)
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    for ax, name in zip(axes.flat, names):
        sns.heatmap(results[name]["confusion_matrix"], annot=True, fmt="d",
                    cbar=False, cmap="Blues", xticklabels=["Ham", "Spam"],
                    yticklabels=["Ham", "Spam"], ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Test-set confusion matrices", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(FIGURE_DIR / "confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def results_table(results):
    rows = []
    for name, value in results.items():
        rows.append({
            "Model": name,
            "Family": value["model_family"],
            "Accuracy": value["accuracy"],
            "Precision (spam)": value["precision_spam"],
            "Recall (spam)": value["recall_spam"],
            "F1 (spam)": value["f1_spam"],
            "Parameters": value["parameters"],
            "Estimated weights (MB, fp32)": value.get("estimated_weights_mb_fp32"),
            "Model load (s)": value.get("model_load_seconds"),
            "Training (s)": value["training_seconds"],
            "Test inference (s)": value["inference_seconds"],
        })
    return pd.DataFrame(rows)


def save_results(results, histories, train_frame, validation_frame, test_frame,
                 best_pretrained, correct, incorrect):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    table = results_table(results)
    table.to_csv(OUTPUT_DIR / "comparison_results.csv", index=False)
    pd.DataFrame([
        {"Model": name, **metrics} for name, metrics in ASSIGNMENT_2_RESULTS.items()
    ]).to_csv(OUTPUT_DIR / "assignment2_reported_metrics.csv", index=False)
    payload = {
        "seed": SEED,
        "split": {"train": len(train_frame), "validation": len(validation_frame),
                  "test": len(test_frame)},
        "labels": {"0": "ham", "1": "spam"},
        "device": str(DEVICE),
        "assignment2_original_split": "stratified 70/10/20",
        "current_split": "stratified 70/15/15",
        "assignment2_reported_results": ASSIGNMENT_2_RESULTS,
        "matched_split_results": results,
        "learning_curves": histories,
        "selected_pretrained_for_error_analysis": best_pretrained,
        "error_analysis_counts": {"correct_examples": len(correct),
                                  "incorrect_examples": len(incorrect)},
    }
    (OUTPUT_DIR / "experiment_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    plot_loss_curves(histories)
    plot_accuracy(results)
    plot_confusion_matrices(results)
    return table


test_frame_global = None


def main():
    global test_frame_global
    seed_everything()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_sms_dataset()
    train_frame, validation_frame, test_frame = split_data(frame)
    test_frame_global = test_frame
    print(f"Dataset: {len(frame)} messages; spam={int(frame.label.sum())}; "
          f"ham={int((frame.label == 0).sum())}")
    print(f"Split sizes: train={len(train_frame)}, validation={len(validation_frame)}, "
          f"test={len(test_frame)}; device={DEVICE}")

    scratch_results, scratch_histories, _ = train_scratch_models(
        train_frame, validation_frame, test_frame)
    pretrained_results, pretrained_histories = train_pretrained_models(
        train_frame, validation_frame, test_frame)
    results = {**scratch_results, **pretrained_results}
    histories = {**scratch_histories, **pretrained_histories}
    best_pretrained, correct, incorrect = choose_error_examples(results)
    table = save_results(results, histories, train_frame, validation_frame,
                         test_frame, best_pretrained, correct, incorrect)
    print("\nMatched-split test results (spam is the positive class):")
    print(table[["Model", "Accuracy", "Precision (spam)", "Recall (spam)",
                 "F1 (spam)", "Training (s)", "Test inference (s)"]].to_string(
                     index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected for error analysis by validation F1: {best_pretrained}")
    print("\nFive correct examples:")
    print(correct[["case_type", "actual", "predicted", "spam_probability", "message"]]
          .to_string(index=False))
    print("\nFive incorrect examples:")
    print(incorrect[["case_type", "actual", "predicted", "spam_probability", "message"]]
          .to_string(index=False))
    print(f"\nSaved results and figures under {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

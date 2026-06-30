from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "data" / "train.csv"
VAL_FILE = BASE_DIR / "data" / "val.csv"
MODEL_NAME = "roberta-base"
OUTPUT_DIR = BASE_DIR / "my-fakenews-model"


def compute_metrics(y_true, y_pred):
    """Compute Accuracy, Precision, Recall, and F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        "f1": float(f1),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
    }


def _import_training_components():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from torch.optim import AdamW
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        return torch, Dataset, DataLoader, AdamW, AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        print("ERROR: Could not import required training dependencies.")
        print("Run these commands from e:/fakenews/backend and retry:")
        print("  pip install -r requirements.txt")
        print(
            "  pip install --upgrade --force-reinstall "
            "torch==2.3.0 torchvision==0.18.0 transformers==4.40.1"
        )
        print(f"\nUnderlying import error: {exc}")
        return None


def train(
    model_name: str = MODEL_NAME,
    epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 512,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    smoke: bool = False,
) -> bool:
    imported = _import_training_components()
    if imported is None:
        return False

    torch, Dataset, DataLoader, AdamW, AutoModelForSequenceClassification, AutoTokenizer = imported

    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        print("ERROR: Dataset files not found. Run prepare_data.py first.")
        return False

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading CSV datasets...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)

    if max_train_samples:
        train_df = train_df.head(max_train_samples)
    if max_val_samples:
        val_df = val_df.head(max_val_samples)

    print(f"Training rows: {len(train_df)} | Validation rows: {len(val_df)}")

    class NewsDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            content = str(self.texts[idx])
            encoding = tokenizer(
                content,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in encoding.items()}
            item["labels"] = self.labels[idx]
            return item

    print("Preparing Dataset objects...")
    train_ds = NewsDataset(train_df["content"].tolist(), train_df["label"].tolist())
    val_ds = NewsDataset(val_df["content"].tolist(), val_df["label"].tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Initializing model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_f1 = -1.0
    best_state = None
    patience = 1 if smoke else 2
    no_improve_epochs = 0
    history = []

    print(f"Training started on {device}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"]
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy().tolist()

                val_preds.extend(preds)
                val_true.extend(labels.numpy().tolist())

        metrics = compute_metrics(val_true, val_preds)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = avg_train_loss
        history.append(metrics)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_f1={metrics['f1']:.4f} | val_acc={metrics['accuracy']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs > patience:
                print("Early stopping: no F1 improvement.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving fine-tuned model to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Save training history
    import pickle
    history_file = BASE_DIR / "training_history.pkl"
    with open(history_file, "wb") as f:
        pickle.dump(history, f)
    print(f"Training history saved to {history_file}")

    print("\nTRAINING COMPLETE. Starting final evaluation...")
    
    # Automatically call evaluate.py
    try:
        from evaluate import evaluate as run_eval
        # Default to full test set evaluation unless limited
        eval_ok = run_eval(
            model_path=OUTPUT_DIR,
            test_file=BASE_DIR / "data" / "test.csv",
            report_path=BASE_DIR / "evaluation_report.txt",
            image_path=BASE_DIR / "confusion_matrix.png",
            max_samples=max_val_samples, # Use max_val_samples as a heuristic for quick eval
        )
        if eval_ok:
            print("\nPERFORMANCE METRICS SUMMARY:")
            print("-" * 60)
            if (BASE_DIR / "evaluation_report.txt").exists():
                with open(BASE_DIR / "evaluation_report.txt", "r") as f:
                    print(f.read())
    except Exception as e:
        print(f"Auto-evaluation failed: {e}")
        print("You can run evaluation manually with: python backend/evaluate.py")

    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VerifyAI fake-news model")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Quick tiny training run for validation")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.smoke:
        if args.max_train_samples is None:
            args.max_train_samples = 128
        if args.max_val_samples is None:
            args.max_val_samples = 64
        if args.epochs == 3:
            args.epochs = 1
        if args.batch_size == 16:
            args.batch_size = 4
        if args.max_length == 512:
            args.max_length = 128

    ok = train(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        smoke=args.smoke,
    )
    raise SystemExit(0 if ok else 1)

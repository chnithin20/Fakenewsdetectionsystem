from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
TEST_FILE = BASE_DIR / "data" / "test.csv"
MODEL_PATH = BASE_DIR / "my-fakenews-model"
IMAGE_PATH = BASE_DIR / "confusion_matrix.png"
REPORT_PATH = BASE_DIR / "evaluation_report.txt"


def _build_classifier(model_path: Path):
    """Create a text-classification pipeline with resilient import handling."""
    try:
        import torch
        from transformers import pipeline
    except Exception as exc:
        print("ERROR: Could not import required ML libraries for evaluation.")
        print("Run these commands from e:/fakenews/backend and try again:")
        print("  pip install -r requirements.txt")
        print(
            "  pip install --upgrade --force-reinstall "
            "torch==2.3.0 torchvision==0.18.0 transformers==4.40.1"
        )
        print(f"\nUnderlying import error: {exc}")
        return None

    try:
        device = 0 if torch.cuda.is_available() else -1
        return pipeline(
            "text-classification",
            model=str(model_path),
            tokenizer=str(model_path),
            device=device,
        )
    except Exception as exc:
        print("ERROR: Pipeline initialization failed.")
        print(f"Check model files under: {model_path}")
        print(f"Details: {exc}")
        return None


def evaluate(
    model_path: Path = MODEL_PATH,
    test_file: Path = TEST_FILE,
    report_path: Path = REPORT_PATH,
    image_path: Path = IMAGE_PATH,
    batch_size: int = 16,
    max_content_chars: int = 2000,
    max_samples: int | None = None,
) -> bool:
    if not model_path.exists():
        print(f"ERROR: Model not found in {model_path}. Run train.py first.")
        return False

    if not test_file.exists():
        print(f"ERROR: Test set {test_file} not found. Run prepare_data.py first.")
        return False

    print("Loading model and test dataset...")
    df_test = pd.read_csv(test_file)

    if max_samples:
        df_test = df_test.head(max_samples)

    classifier = _build_classifier(model_path)
    if classifier is None:
        return False

    print(f"Running inference on {len(df_test)} test samples...")
    contents = df_test["content"].astype(str).tolist()
    contents = [c[:max_content_chars] for c in contents]

    results = classifier(contents, batch_size=batch_size, truncation=True)

    y_true = df_test["label"].tolist()
    y_pred = [1 if r["label"] == "LABEL_1" else 0 for r in results]
    y_probs = [r["score"] if r["label"] == "LABEL_1" else (1 - r["score"]) for r in results]

    print("Generating classification report...")
    report = classification_report(y_true, y_pred, target_names=["REAL", "FAKE"])
    print("\n" + report)

    with report_path.open("w", encoding="utf-8") as report_file:
        report_file.write("FAKE NEWS DETECTION EVALUATION REPORT\n")
        report_file.write("=" * 40 + "\n")
        report_file.write(report)

    # Save numeric evaluation data to PKL
    import pickle
    results_pkl = report_path.with_suffix(".pkl")
    with open(results_pkl, "wb") as f:
        pickle.dump({
            "y_true": y_true,
            "y_pred": y_pred,
            "y_probs": y_probs,
            "report_dict": report
        }, f)
    print(f"Evaluation metrics saved to {results_pkl}")

    print("Plotting confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix - Fake News Detection")
    plt.savefig(image_path)
    print(f"Confusion matrix saved to {image_path}")

    print("\nAnalyzing most confident errors...")
    errors = []
    for i, truth in enumerate(y_true):
        pred = y_pred[i]
        if truth != pred:
            confidence = y_probs[i] if pred == 1 else (1 - y_probs[i])
            errors.append(
                {
                    "content": contents[i][:150],
                    "actual": "FAKE" if truth == 1 else "REAL",
                    "predicted": "FAKE" if pred == 1 else "REAL",
                    "confidence": confidence,
                }
            )

    errors.sort(key=lambda x: x["confidence"], reverse=True)

    print("\nTOP 10 CONFIDENTLY WRONG PREDICTIONS:")
    print("-" * 60)
    for i, err in enumerate(errors[:10], start=1):
        print(
            f"{i}. Predicted {err['predicted']} | "
            f"Actual {err['actual']} | Conf {err['confidence']:.4f}"
        )
        print(f"Content: {err['content']}...")
        print("-" * 60)

    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VerifyAI fake-news model")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--test-file", default=str(TEST_FILE))
    parser.add_argument("--report-path", default=str(REPORT_PATH))
    parser.add_argument("--image-path", default=str(IMAGE_PATH))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-content-chars", type=int, default=2000)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ok = evaluate(
        model_path=Path(args.model_path),
        test_file=Path(args.test_file),
        report_path=Path(args.report_path),
        image_path=Path(args.image_path),
        batch_size=args.batch_size,
        max_content_chars=args.max_content_chars,
        max_samples=args.max_samples,
    )
    raise SystemExit(0 if ok else 1)

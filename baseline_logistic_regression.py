import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

# =========================
# 1. PATHS AND SETTINGS
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "HAM10000"

METADATA_FILE = DATA_DIR / "GroundTruth.csv"
IMG_DIR = DATA_DIR / "images"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 64            # try 32 if your laptop is slow
USE_GRAYSCALE = False    # RGB usually performs better
MAX_IMAGES = 5000        # reduce to 1000 if needed
RANDOM_STATE = 42

# Threshold tuning: lower than 0.5 helps catch more malignant cases
THRESHOLD = 0.35

# =========================
# 2. CHECK FILES EXIST
# =========================
print("\nChecking dataset paths...\n")
print("Metadata file:", METADATA_FILE)
print("Exists:", METADATA_FILE.exists())

print("\nImage folder:", IMG_DIR)
print("Exists:", IMG_DIR.exists())

if not METADATA_FILE.exists():
    raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

if not IMG_DIR.exists():
    raise FileNotFoundError(f"Image folder not found: {IMG_DIR}")

# =========================
# 3. LOAD METADATA
# =========================
print("\nLoading metadata...")
df = pd.read_csv(METADATA_FILE)
print("Metadata shape:", df.shape)
print(df.head())

# =========================
# 4. CONVERT ONE-HOT LABELS TO SINGLE DIAGNOSIS LABEL
# =========================
label_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

missing_cols = [col for col in label_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected label columns: {missing_cols}")

df["dx"] = df[label_cols].idxmax(axis=1)

print("\nDiagnosis counts:")
print(df["dx"].value_counts())

# malignant = MEL, BCC, AKIEC
malignant_classes = ["MEL", "BCC", "AKIEC"]
df["target"] = df["dx"].apply(lambda x: 1 if x in malignant_classes else 0)

print("\nBinary target counts:")
print(df["target"].value_counts())
print("0 = benign, 1 = malignant")

# =========================
# 5. CREATE IMAGE PATHS
# =========================
df["image_path"] = df["image"].apply(lambda x: str(IMG_DIR / f"{x}.jpg"))
df = df[df["image_path"].apply(os.path.exists)].copy()

print("\nRows after matching image paths:", len(df))
print(df[["image", "dx", "target", "image_path"]].head())

# =========================
# 6. USE BALANCED SUBSET
# =========================
if MAX_IMAGES is not None:
    print(f"\nUsing a balanced subset of up to {MAX_IMAGES} images...")

    benign_df = df[df["target"] == 0]
    malignant_df = df[df["target"] == 1]

    benign_n = min(len(benign_df), MAX_IMAGES // 2)
    malignant_n = min(len(malignant_df), MAX_IMAGES // 2)

    df = pd.concat([
        benign_df.sample(n=benign_n, random_state=RANDOM_STATE),
        malignant_df.sample(n=malignant_n, random_state=RANDOM_STATE)
    ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("Final dataset size:", len(df))
print(df["target"].value_counts())

# =========================
# 7. SAVE CLASS DISTRIBUTION PLOT
# =========================
print("\nSaving class distribution plot...")
class_counts = df["target"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
class_counts.plot(kind="bar")
plt.xticks([0, 1], ["Benign (0)", "Malignant (1)"], rotation=0)
plt.ylabel("Count")
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution.png")
plt.close()

# =========================
# 8. SAVE SAMPLE IMAGES PLOT
# =========================
print("Saving sample images plot...")
sample_df = df.sample(min(6, len(df)), random_state=RANDOM_STATE)

fig, axes = plt.subplots(2, 3, figsize=(10, 7))
axes = axes.flatten()

for ax in axes:
    ax.axis("off")

for ax, (_, row) in zip(axes, sample_df.iterrows()):
    img = Image.open(row["image_path"])
    ax.imshow(img)
    ax.set_title(f'dx={row["dx"]}, target={row["target"]}')
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sample_images.png")
plt.close()

# =========================
# 9. PREPROCESS FUNCTION
# =========================
def preprocess_image(image_path, img_size=48, use_grayscale=False):
    img = Image.open(image_path)

    if use_grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    img = img.resize((img_size, img_size))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array.flatten()

# =========================
# 10. BUILD X AND y
# =========================
print("\nPreprocessing images... This may take a few minutes.")
X = []
y = []

for _, row in df.iterrows():
    features = preprocess_image(
        row["image_path"],
        img_size=IMG_SIZE,
        use_grayscale=USE_GRAYSCALE
    )
    X.append(features)
    y.append(row["target"])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# 11. TRAIN / VAL / TEST SPLIT
# =========================
print("\nSplitting data into train, validation, and test sets...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("Train shape:", X_train.shape, y_train.shape)
print("Validation shape:", X_val.shape, y_val.shape)
print("Test shape:", X_test.shape, y_test.shape)

# =========================
# 12. MODEL PIPELINE
# =========================
print("\nTraining Logistic Regression pipeline...")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=75, random_state=RANDOM_STATE)),
    ("logreg", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="saga",
        C=1.0
    ))
])

model.fit(X_train, y_train)
print("Model training complete.")

# =========================
# 13. VALIDATION EVALUATION
# =========================
print("\nEvaluating on validation set...")
y_val_prob = model.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_prob >= THRESHOLD).astype(int)

val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred, zero_division=0)
val_recall = recall_score(y_val, y_val_pred, zero_division=0)
val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
val_auc = roc_auc_score(y_val, y_val_prob)

print("\nValidation Results")
print("------------------")
print("Threshold:", THRESHOLD)
print("Accuracy :", val_accuracy)
print("Precision:", val_precision)
print("Recall   :", val_recall)
print("F1-score :", val_f1)
print("ROC-AUC  :", val_auc)

# =========================
# 14. TEST EVALUATION
# =========================
print("\nEvaluating on test set...")
y_test_prob = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= THRESHOLD).astype(int)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
test_auc = roc_auc_score(y_test, y_test_prob)

print("\nTest Results")
print("------------")
print("Threshold:", THRESHOLD)
print("Accuracy :", test_accuracy)
print("Precision:", test_precision)
print("Recall   :", test_recall)
print("F1-score :", test_f1)
print("ROC-AUC  :", test_auc)

# =========================
# 15. CLASSIFICATION REPORT
# =========================
print("\nClassification Report (Test Set):")
report = classification_report(
    y_test,
    y_test_pred,
    target_names=["Benign", "Malignant"],
    zero_division=0
)
print(report)

with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

# =========================
# 16. CONFUSION MATRIX
# =========================
print("Saving confusion matrix plot...")
cm = confusion_matrix(y_test, y_test_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - Test Set (threshold={THRESHOLD})")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.close()

# =========================
# 17. ROC CURVE
# =========================
print("Saving ROC curve plot...")
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {test_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Test Set")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png")
plt.close()

# =========================
# 18. SAVE RESULTS TABLE
# =========================
print("Saving metrics CSV...")
results_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
    "Validation": [val_accuracy, val_precision, val_recall, val_f1, val_auc],
    "Test": [test_accuracy, test_precision, test_recall, test_f1, test_auc]
})

results_df.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False)
print(results_df)

# =========================
# 19. SAVE TEST PREDICTIONS
# =========================
print("Saving test predictions CSV...")
predictions_df = pd.DataFrame({
    "true_label": y_test,
    "predicted_label": y_test_pred,
    "predicted_probability_malignant": y_test_prob
})
predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

# =========================
# 20. DONE
# =========================
print("\nAll done.")
print(f"Check your outputs folder here:\n{OUTPUT_DIR}")
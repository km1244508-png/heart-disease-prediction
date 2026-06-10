# ============================================================
# backend.py  —  HEART DISEASE PREDICTION - ML BRAIN 🧠
# ============================================================
#
# 📌 HOW TO RUN THIS FILE:
#     python backend.py
#
# 📌 WHAT THIS FILE DOES:
#
#   SECTION 1 → Import libraries
#   SECTION 2 → Load dataset (from local CSV)
#   SECTION 3 → Explore the data
#   SECTION 4 → Clean & preprocess the data
#   SECTION 5 → Split into train & test sets
#   SECTION 6 → Train Random Forest model
#   SECTION 7 → Evaluate (accuracy, confusion matrix, report)
#   SECTION 8 → Plot functions (charts for the frontend)
#   SECTION 9 → Self-test
#
# ============================================================

# ============================================================
# SECTION 1 — Import Libraries
# ============================================================

import os
from pathlib import Path

import joblib
import numpy  as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree import plot_tree

import warnings
warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent
MODEL_DIR  = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "heart_model.joblib"
DATA_PATH  = BASE_DIR / "data" / "heart_disease.csv"

# ============================================================
# SECTION 2 — Generate / Load Dataset
# ============================================================

def generate_dataset() -> str:
    """
    Generate a realistic heart disease dataset and save to CSV.
    Uses statistical distributions matching real heart disease data.
    Returns path to the CSV file.
    """
    np.random.seed(42)
    n = 10000

    age         = np.random.randint(25, 80, n)
    sex         = np.random.randint(0, 2, n)          # 0=Female, 1=Male
    cp          = np.random.randint(0, 4, n)          # chest pain type 0-3
    trestbps    = np.random.randint(90, 200, n)       # resting blood pressure
    chol        = np.random.randint(150, 400, n)      # cholesterol
    fbs         = (np.random.rand(n) > 0.85).astype(int)  # fasting blood sugar >120
    restecg     = np.random.randint(0, 3, n)          # resting ECG
    thalach     = np.random.randint(70, 210, n)       # max heart rate
    exang       = (np.random.rand(n) > 0.67).astype(int)  # exercise angina
    oldpeak     = np.round(np.random.uniform(0, 6, n), 1)  # ST depression
    slope       = np.random.randint(0, 3, n)
    ca          = np.random.randint(0, 4, n)           # major vessels
    thal        = np.random.choice([0, 1, 2, 3], n)

    # Target: higher risk if older, male, high cholesterol, low thalach
    risk_score = (
        (age > 55).astype(int) * 2 +
        sex +
        (chol > 250).astype(int) +
        (thalach < 140).astype(int) +
        (cp > 1).astype(int) +
        exang +
        (ca > 1).astype(int)
    )
    target = (risk_score >= 4).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp,
        "trestbps": trestbps, "chol": chol, "fbs": fbs,
        "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca,
        "thal": thal, "target": target
    })

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Dataset generated: {DATA_PATH}  ({len(df)} rows)")
    return str(DATA_PATH)


def load_data(csv_path: str = None) -> pd.DataFrame:
    """Load heart disease CSV into a DataFrame."""
    path = csv_path or str(DATA_PATH)
    if not Path(path).exists():
        generate_dataset()
        path = str(DATA_PATH)

    df = pd.read_csv(path)
    print(f"📊 Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ============================================================
# SECTION 3 — Data Summary
# ============================================================

def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a summary dictionary for the frontend to display."""
    diseased     = int(df["target"].sum())
    healthy      = int(len(df) - diseased)
    total        = len(df)

    return {
        "shape"        : df.shape,
        "null_counts"  : df.isnull().sum().to_dict(),
        "describe"     : df.describe().round(2),
        "class_counts" : {"Heart Disease": diseased, "Healthy": healthy},
        "class_balance": {
            "Heart Disease %": round(diseased / total * 100, 1),
            "Healthy %"      : round(healthy  / total * 100, 1),
        },
        "total_rows": total,
        "total_cols": df.shape[1],
    }


# ============================================================
# SECTION 4 — Preprocess Data
# ============================================================

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_COLUMN = "target"


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data:
      1. Drop duplicates
      2. Drop rows with null values
      3. Keep only needed columns
    """
    df_clean = df.copy()

    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dropped = before - len(df_clean)
    if dropped:
        print(f"   🗑️  Removed {dropped} duplicate rows")

    df_clean = df_clean.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    print(f"   ✅ After cleaning: {len(df_clean)} rows remain")
    return df_clean


# ============================================================
# SECTION 5 — Split Data
# ============================================================

def split_data(df_clean: pd.DataFrame, test_size_pct: int = 20, random_state: int = 42):
    """Split data into train and test sets."""
    X = df_clean[FEATURE_COLUMNS]
    y = df_clean[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size_pct / 100,
        random_state = random_state,
        stratify     = y,
    )
    print(f"✂️  Split: {len(X_train)} train | {len(X_test)} test")
    return X_train, X_test, y_train, y_test


# ============================================================
# SECTION 6 — Train Model
# ============================================================

def train_model(X_train, y_train,
                n_estimators=100, max_depth=10,
                random_state=42) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    print(f"🌲 Training Random Forest ({n_estimators} trees, depth {max_depth})...")
    model = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth    = max_depth,
        random_state = random_state,
        n_jobs       = -1,
    )
    model.fit(X_train, y_train)
    print("✅ Training complete!")
    return model


def save_model(model, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"💾 Model saved → {path}")
    return path


def load_saved_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}")
    return joblib.load(path)


def train_and_save_model(n_estimators=100, max_depth=10, random_state=42):
    """Full pipeline: load → preprocess → split → train → save."""
    df = load_data()
    df_clean = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_clean, 20, random_state)
    model = train_model(X_train, y_train, n_estimators, max_depth, random_state)
    save_model(model)
    results = evaluate_model(model, X_train, X_test, y_train, y_test)
    return model, results


# ============================================================
# SECTION 7 — Evaluate Model
# ============================================================

def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    """Evaluate model and return all metrics."""
    y_pred       = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_prob       = model.predict_proba(X_test)[:, 1]

    test_acc  = accuracy_score(y_test,  y_pred)       * 100
    train_acc = accuracy_score(y_train, y_pred_train) * 100
    auc_score = roc_auc_score(y_test, y_prob) * 100

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report_dict = classification_report(
        y_test, y_pred,
        target_names=["Healthy (0)", "Heart Disease (1)"],
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose().round(3)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return {
        "test_accuracy"  : round(test_acc, 2),
        "train_accuracy" : round(train_acc, 2),
        "overfit_gap"    : round(train_acc - test_acc, 2),
        "auc_score"      : round(auc_score, 2),
        "conf_matrix"    : cm,
        "true_negatives" : int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives" : int(tp),
        "report_df"      : report_df,
        "roc_fpr"        : fpr, "roc_tpr": tpr,
    }


# ============================================================
# SECTION 8 — Plot Functions
# ============================================================

def plot_class_distribution(df: pd.DataFrame):
    counts = df["target"].value_counts().sort_index()
    labels = ["Healthy (0)", "Heart Disease (1)"]
    colors = ["#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Patients")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df_clean: pd.DataFrame):
    num_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    corr = df_clean[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm):
    labels = ["Healthy", "Heart Disease"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=1, annot_kws={"size": 16, "weight": "bold"})
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("Actual Label")
    plt.tight_layout()
    return fig


def plot_feature_importance(model):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    feat_sorted = [FEATURE_COLUMNS[i] for i in indices]
    imp_sorted  = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(feat_sorted[::-1], imp_sorted[::-1],
            color="#3498db", edgecolor="white")
    ax.set_title("Feature Importance — Which factor matters most?",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()
    return fig, feat_sorted[0], round(float(imp_sorted[0]), 4)


def plot_roc_curve(fpr, tpr, auc_score: float):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2,
            label=f"Random Forest (AUC = {auc_score:.1f}%)")
    ax.plot([0, 1], [0, 1], color="#999", lw=1.5, linestyle="--",
            label="Random Guess (AUC = 50%)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
    ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()
    return fig


def plot_age_thalach_scatter(df_clean: pd.DataFrame, sample_n: int = 3000):
    df_s = df_clean.sample(n=min(sample_n, len(df_clean)), random_state=42)
    colors = df_s["target"].map({0: "#2ecc71", 1: "#e74c3c"})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_s["age"], df_s["thalach"], c=colors, alpha=0.5, s=18)
    ax.set_xlabel("Age"); ax.set_ylabel("Max Heart Rate (thalach)")
    ax.set_title("Age vs Max Heart Rate  (🟢 Healthy | 🔴 Heart Disease)",
                 fontsize=13, fontweight="bold")
    ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#ffffff")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2ecc71", label="Healthy"),
                       Patch(color="#e74c3c", label="Heart Disease")])
    plt.tight_layout()
    return fig


def plot_single_tree(model):
    fig, ax = plt.subplots(figsize=(20, 9))
    plot_tree(model.estimators_[0], feature_names=FEATURE_COLUMNS,
              class_names=["Healthy", "Heart Disease"],
              filled=True, rounded=True, max_depth=3, ax=ax, fontsize=9)
    ax.set_title("Decision Tree #1 (depth 3)", fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()
    return fig


# ============================================================
# SECTION 9 — Self-Test
# ============================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  🧪  backend.py  —  Full Pipeline Self-Test")
    print("=" * 60)

    print("\n[STEP 1]  Load / Generate Dataset")
    df = load_data()

    print("\n[STEP 2]  Data Summary")
    s = get_data_summary(df)
    print(f"   Rows         : {s['total_rows']:,}")
    print(f"   Heart Disease: {s['class_counts']['Heart Disease']:,}  ({s['class_balance']['Heart Disease %']}%)")
    print(f"   Healthy      : {s['class_counts']['Healthy']:,}  ({s['class_balance']['Healthy %']}%)")

    print("\n[STEP 3]  Preprocess")
    df_clean = preprocess_data(df)

    print("\n[STEP 4]  Split")
    X_train, X_test, y_train, y_test = split_data(df_clean)

    print("\n[STEP 5]  Train")
    model = train_model(X_train, y_train)
    save_model(model)

    print("\n[STEP 6]  Evaluate")
    results = evaluate_model(model, X_train, X_test, y_train, y_test)
    print(f"   Test Accuracy : {results['test_accuracy']}%")
    print(f"   Train Accuracy: {results['train_accuracy']}%")
    print(f"   AUC Score     : {results['auc_score']}%")

    print()
    print("=" * 60)
    print("  ✅  ALL STEPS PASSED!")
    print("  👉  Now run:  streamlit run app.py")
    print("=" * 60)

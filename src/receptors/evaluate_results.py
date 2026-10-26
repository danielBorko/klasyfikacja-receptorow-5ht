"""
Skrypt porównawczy (Evaluation Script) dla modeli  (A: biochemiczny, B: ESM-2).
Zawiera szczegółową analizę pomyłek klasyfikacji w podziale na gatunki i typy receptorów.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from splits import get_group_splits, get_stratified_splits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

try:
    from receptors.splits import get_group_splits
except ImportError:
    from splits import get_group_splits

BASE_DIR = Path(__file__).resolve().parents[2]
BIO_PATH = BASE_DIR / "dane" / "processed" / "biochemical_features.csv"
ESM_PATH = BASE_DIR / "dane" / "processed" / "esm_embeddings.pt"


def load_biochem_data(path=BIO_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Brak pliku {path}.")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    y = df["label"]
    species = df["species"] if "species" in df.columns else (df["raw"] if "raw" in df.columns else df.iloc[:, 2])
    X = df.drop(columns=["sample_id", "gene", "species", "raw", "label", "uniprot_id"], errors="ignore").fillna(0)
    X.index = df["sample_id"] if "sample_id" in df.columns else df.index
    return X, y, species


def load_esm_data(path=ESM_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Brak pliku {path}.")
    df_esm = torch.load(path, weights_only=False)
    if isinstance(df_esm, pd.DataFrame):
        df_esm.columns = df_esm.columns.str.strip()
        y = df_esm["label"]
        species = df_esm["species"] if "species" in df_esm.columns else (
            df_esm["raw"] if "raw" in df_esm.columns else df_esm.iloc[:, 1])
        X = pd.DataFrame(df_esm["embedding"].tolist(), index=df_esm["sample_id"])
    else:
        raise ValueError("Załadowany obiekt ESM nie jest ramką danych Pandas!")
    return X, y, species





def evaluate_single_model(X, y, species, model_name="Model", mode="stratified"):
    df_temp = pd.DataFrame({"raw": species, "label": y})


    if mode == "group":
        splits = list(get_group_splits(df_temp, n_splits=5))
    else:
        splits = list(get_stratified_splits(df_temp, n_splits=5))

    macro_f1s, bal_accs, accs = [], [], []
    all_y_true, all_y_pred = [], []
    error_records = []

    for train_idx, test_idx in splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        sp_test = species.iloc[test_idx]
        sample_ids = X.index[test_idx]

        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        macro_f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        bal_accs.append(balanced_accuracy_score(y_test, y_pred))
        accs.append(accuracy_score(y_test, y_pred))

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        for s_id, sp, true_l, pred_l in zip(sample_ids, sp_test, y_test, y_pred):
            if true_l != pred_l:
                error_records.append({
                    "sample_id": s_id,
                    "species": sp,
                    "true_label": true_l,
                    "pred_label": pred_l
                })

    metrics = {
        "Model": f"{model_name} ({mode.upper()})",
        "Macro-F1": f"{np.mean(macro_f1s):.3f} ± {np.std(macro_f1s):.3f}",
        "Balanced Acc": f"{np.mean(bal_accs):.3f} ± {np.std(bal_accs):.3f}",
        "Accuracy": f"{np.mean(accs):.3f} ± {np.std(accs):.3f}"
    }

    df_errors = pd.DataFrame(error_records)
    return metrics, all_y_true, all_y_pred, df_errors


def run_full_evaluation():
    reports_dir = BASE_DIR / "reports" / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    results = []

    print("=== EWALUACJA MODELI + SZCZEGÓŁOWA ANALIZA BŁĘDÓW ===")

    # 1. Model A
    print("\n" + "=" * 50)
    print("MODEL A: Cechy Biochemiczne - Szczegóły pomyłek")
    print("=" * 50)
    X_bio, y_bio, sp_bio = load_biochem_data()
    m_bio, y_true_bio, y_pred_bio, df_err_bio = evaluate_single_model(X_bio, y_bio, sp_bio,
                                                                      model_name="Model A (Biochemiczny)")
    results.append(m_bio)
    plot_confusion_matrix(y_true_bio, y_pred_bio, reports_dir / "cm_model_A.png", "Model A: Cechy Biochemiczne")

    if not df_err_bio.empty:
        print("\nWszystkie błędne klasyfikacje (Model A):")
        print(df_err_bio.to_string(index=False))
    else:
        print("Brak błędów!")

    # 2. Model B
    print("\n" + "=" * 50)
    print("MODEL B: Embeddingi ESM-2 - Szczegóły pomyłek")
    print("=" * 50)
    X_esm, y_esm, sp_esm = load_esm_data()
    m_esm, y_true_esm, y_pred_esm, df_err_esm = evaluate_single_model(X_esm, y_esm, sp_esm,
                                                                      model_name="Model B (ESM-2)")
    results.append(m_esm)
    plot_confusion_matrix(y_true_esm, y_pred_esm, reports_dir / "cm_model_B.png", "Model B: ESM-2")

    if not df_err_esm.empty:
        print("\nWszystkie błędne klasyfikacje (Model B):")
        print(df_err_esm.to_string(index=False))
    else:
        print("Brak błędów!")

    if results:
        df_results = pd.DataFrame(results)
        print("\n" + "=" * 60)
        print("ZEBRANE WYNIKI DO TABELI PORÓWNAWCZEJ W PRACY:")
        print("=" * 60)
        print(df_results.to_string(index=False))
        print("=" * 60)


def plot_confusion_matrix(y_true, y_pred, save_path, title):
    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Macierz pomyłek – {title}", fontsize=12, weight="bold")
    plt.xlabel("Przewidziana etykieta (Predicted)")
    plt.ylabel("Prawdziwa etykieta (True)")
    plt.tight_layout()
    path_str = str(Path(save_path).resolve())
    plt.savefig(path_str, dpi=300)
    plt.close()


if __name__ == "__main__":
    run_full_evaluation()
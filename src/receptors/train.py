"""
Moduł trenowania i ewaluacji modeli zgodnie z oficjalnym protokołem eksperymentalnym.
Obsługuje walidację Stratified 5-fold oraz GroupKFold po gatunkach[cite: 8, 17].
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from splits import get_stratified_splits, get_group_splits

def evaluate_model_pipeline(features_path: str, model_type: str = "biochemical", validation_mode: str = "group"):
    """
    Trenuje i ewaluuje model przy użyciu wskazanego protokołu walidacji.
    Zapisuje wytrenowany model oraz raport z metrykami.
    """
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Nie znaleziono pliku cech pod ścieżką: {features_path}")

    # Wczytanie danych
    df = pd.read_parquet(features_path)

    # Przygotowanie macierzy cech X i etykiet y
    X = df.drop(columns=["sample_id", "gene", "raw", "label", "uniprot_id"], errors="ignore")
    y = df["label"]
    X = X.fillna(0)

    # Wybór protokołu walidacji
    if validation_mode == "group":
        print("Uruchamianie walidacji: GroupKFold po gatunkach (główny protokół ewolucyjny)[cite: 8, 17]...")
        splits = list(get_group_splits(df, n_splits=5))
    else:
        print("Uruchamianie walidacji: Stratified 5-Fold (protokół pomocniczy)[cite: 8, 18]...")
        splits = list(get_stratified_splits(df, n_splits=5))

    macro_f1_list = []
    balanced_acc_list = []
    accuracy_list = []

    best_model = None
    best_score = -1

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Inicjalizacja klasyfikatora (np. Random Forest)
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Obliczanie metryk
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        macro_f1_list.append(macro_f1)
        balanced_acc_list.append(bal_acc)
        accuracy_list.append(acc)

        # model z najlepszym wynikiem Macro-F1
        if macro_f1 > best_score:
            best_score = macro_f1
            best_model = model

        print(f" -> Fold {fold+1}: Macro-F1 = {macro_f1:.3f} | Balanced Acc = {bal_acc:.3f} | Accuracy = {acc:.3f}")

    # Podsumowanie wyników (średnia ± odchylenie standardowe)
    print("\n--------------------------------------------------")
    print(f" WYNIKI KOŃCOWE ({model_type.upper()} | Tryb: {validation_mode})")
    print(f" Średnie Macro-F1:       {np.mean(macro_f1_list):.3f} ± {np.std(macro_f1_list):.3f}[cite: 18, 19]")
    print(f" Średnia Balanced Acc:  {np.mean(balanced_acc_list):.3f} ± {np.std(balanced_acc_list):.3f}[cite: 18, 19]")
    print(f" Średnia Accuracy:      {np.mean(accuracy_list):.3f} ± {np.std(accuracy_list):.3f}")
    print("--------------------------------------------------")

    # Zapis najlepszego modelu na dysk
    os.makedirs("models", exist_ok=True)
    model_output_path = f"models/best_model_{model_type}_{validation_mode}.pkl"
    joblib.dump(best_model, model_output_path)
    print(f"Zapisano najlepszy model do: {model_output_path}")

    return best_model
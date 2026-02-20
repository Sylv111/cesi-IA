"""
Script de comparaison de modeles de classification sur general_data_merged.csv.

Fonctionnalites:
- Chargement des donnees
- Preparation (valeurs manquantes, encodage categoriel, normalisation)
- Separation X (features) / y (cible)
- Entrainement de plusieurs modeles
- Comparaison des performances (confusion matrix, precision/recall/F1, ROC/AUC,
  temps d'entrainement et de prediction)

Le script sauvegarde les resultats dans le dossier Output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42


def build_models() -> dict[str, object]:
    """Retourne les modeles a comparer."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Perceptron": Perceptron(max_iter=2000, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1),
    }


def detect_target_column(df: pd.DataFrame, requested_target: str | None) -> str:
    """Determine la colonne cible."""
    if requested_target:
        if requested_target not in df.columns:
            raise ValueError(f"La colonne cible '{requested_target}' est introuvable dans le dataset.")
        return requested_target

    if "Attrition" in df.columns:
        return "Attrition"

    raise ValueError("Aucune cible fournie. Utilisez --target pour preciser la colonne cible.")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construit le preprocessing pour numeriques et categorielles."""
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def score_for_roc(model: Pipeline, X_test: pd.DataFrame) -> np.ndarray:
    """Retourne un score continu pour ROC/AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        decision_values = model.decision_function(X_test)
        min_val = np.min(decision_values)
        max_val = np.max(decision_values)
        if max_val == min_val:
            return np.zeros_like(decision_values, dtype=float)
        return (decision_values - min_val) / (max_val - min_val)

    return model.predict(X_test)


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    preprocessor: ColumnTransformer,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray, float]]]:
    """Entraine et evalue tous les modeles."""
    models = build_models()

    rows: list[dict[str, float | str]] = []
    confusion_matrices: dict[str, np.ndarray] = {}
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        start_train = perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = perf_counter() - start_train

        start_pred = perf_counter()
        y_pred = pipeline.predict(X_test)
        pred_time = perf_counter() - start_pred

        y_score = score_for_roc(pipeline, X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_score)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_score)

        confusion_matrices[model_name] = cm
        roc_data[model_name] = (fpr, tpr, auc)

        rows.append(
            {
                "modele": model_name,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": auc,
                "train_time_sec": train_time,
                "predict_time_sec": pred_time,
            }
        )

    results = pd.DataFrame(rows).sort_values(by="f1", ascending=False).reset_index(drop=True)
    return results, confusion_matrices, roc_data


def save_confusion_matrices(
    confusion_matrices: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Sauvegarde toutes les matrices de confusion dans une image."""
    n_models = len(confusion_matrices)
    cols = 3
    rows = int(np.ceil(n_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(model_name)

    for idx in range(len(confusion_matrices), len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_roc_curves(
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    output_path: Path,
) -> None:
    """Sauvegarde les courbes ROC de tous les modeles."""
    plt.figure(figsize=(8, 6))

    for model_name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Courbes ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparaison de modeles de classification pour general_data_merged.csv"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Chemin du CSV d'entree (defaut: Output/general_data_merged.csv)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Nom de la colonne cible (defaut: Attrition si presente)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion du jeu de test (defaut: 0.2)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    output_dir = project_dir / "Output"
    output_dir.mkdir(exist_ok=True)

    input_path = Path(args.input) if args.input else (output_dir / "general_data_merged.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {input_path}")

    print(f"Lecture du fichier: {input_path}")
    df = pd.read_csv(input_path)

    target_col = detect_target_column(df, args.target)
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    # Encodage binaire de la cible pour metrics et ROC
    y = LabelEncoder().fit_transform(y_raw.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)

    print(f"Cible: {target_col}")
    print(f"X: {X.shape}, y: {y.shape}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    results, confusion_matrices, roc_data = evaluate_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
    )

    metrics_csv_path = output_dir / "general_data_merged_model_comparison.csv"
    results.to_csv(metrics_csv_path, index=False)

    cm_image_path = output_dir / "general_data_merged_confusion_matrices.png"
    save_confusion_matrices(confusion_matrices, cm_image_path)

    roc_image_path = output_dir / "general_data_merged_roc_curves.png"
    save_roc_curves(roc_data, roc_image_path)

    cm_json_path = output_dir / "general_data_merged_confusion_matrices.json"
    cm_json_data = {name: matrix.tolist() for name, matrix in confusion_matrices.items()}
    cm_json_path.write_text(json.dumps(cm_json_data, indent=2), encoding="utf-8")

    print("\nResultats (tries par F1):")
    print(results.to_string(index=False))
    print(f"\nCSV metriques: {metrics_csv_path}")
    print(f"Matrices de confusion (image): {cm_image_path}")
    print(f"Courbes ROC (image): {roc_image_path}")
    print(f"Matrices de confusion (json): {cm_json_path}")


if __name__ == "__main__":
    main()

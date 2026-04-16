from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_tsv_dataset(path: Path):
    data = np.loadtxt(path, delimiter="\t")
    y = data[:, 0].astype(int)
    X = data[:, 1:].astype(np.float64)
    return X, y


def load_sensitive(path: Path):
    return np.loadtxt(path, delimiter=",").astype(int)


def check_finite(name: str, X: np.ndarray):
    num_nan = np.isnan(X).sum()
    num_inf = np.isinf(X).sum()
    print(f"{name}: shape={X.shape}, nan={num_nan}, inf={num_inf}, min={np.min(X):.6f}, max={np.max(X):.6f}")

    if num_nan > 0 or num_inf > 0:
        raise ValueError(f"{name} contains NaN or Inf values.")


def demographic_parity_difference(y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    group0 = y_pred[sensitive == 0]
    group1 = y_pred[sensitive == 1]

    rate0 = np.mean(group0 == 1)
    rate1 = np.mean(group1 == 1)

    return abs(rate1 - rate0)


def equalized_odds_difference(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    # True Positive Rate for each group
    group0_pos = (sensitive == 0) & (y_true == 1)
    group1_pos = (sensitive == 1) & (y_true == 1)

    # False Positive Rate for each group
    group0_neg = (sensitive == 0) & (y_true == 0)
    group1_neg = (sensitive == 1) & (y_true == 0)

    def safe_mean(arr):
        if np.sum(arr) == 0:
            return 0.0
        return np.mean(arr)

    group0_tpr = safe_mean(y_pred[group0_pos] == 1)
    group1_tpr = safe_mean(y_pred[group1_pos] == 1)

    group0_fpr = safe_mean(y_pred[group0_neg] == 1)
    group1_fpr = safe_mean(y_pred[group1_neg] == 1)

    return max(abs(group1_tpr - group0_tpr), abs(group1_fpr - group0_fpr))


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"

    X_train, y_train = load_tsv_dataset(output_dir / "train_processed.tsv")
    X_val, y_val = load_tsv_dataset(output_dir / "val_processed.tsv")
    X_test, y_test = load_tsv_dataset(output_dir / "test_processed.tsv")

    a_train = load_sensitive(output_dir / "train_sensitive.csv")
    a_val = load_sensitive(output_dir / "val_sensitive.csv")
    a_test = load_sensitive(output_dir / "test_sensitive.csv")

    # Sanity checks
    check_finite("X_train", X_train)
    check_finite("X_val", X_val)
    check_finite("X_test", X_test)

    print(f"y_train unique: {np.unique(y_train)}")
    print(f"y_val unique:   {np.unique(y_val)}")
    print(f"y_test unique:  {np.unique(y_test)}")
    print(f"a_train unique: {np.unique(a_train)}")
    print(f"a_val unique:   {np.unique(a_val)}")
    print(f"a_test unique:  {np.unique(a_test)}")

    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_dp = demographic_parity_difference(y_train_pred, a_train)
    val_dp = demographic_parity_difference(y_val_pred, a_val)
    test_dp = demographic_parity_difference(y_test_pred, a_test)

    train_eod = equalized_odds_difference(y_train, y_train_pred, a_train)
    val_eod = equalized_odds_difference(y_val, y_val_pred, a_val)
    test_eod = equalized_odds_difference(y_test, y_test_pred, a_test)

    print("\n=== Baseline Logistic Regression Results ===")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy:   {val_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print()
    print(f"Train Demographic Parity Difference: {train_dp:.4f}")
    print(f"Val Demographic Parity Difference:   {val_dp:.4f}")
    print(f"Test Demographic Parity Difference:  {test_dp:.4f}")
    print()
    print(f"Train Equalized Odds Difference: {train_eod:.4f}")
    print(f"Val Equalized Odds Difference:   {val_eod:.4f}")
    print(f"Test Equalized Odds Difference:  {test_eod:.4f}")

    results_path = output_dir / "baseline_results.txt"
    with open(results_path, "w") as f:
        f.write("=== Baseline Logistic Regression Results ===\n")
        f.write(f"Train Accuracy: {train_acc:.4f}\n")
        f.write(f"Val Accuracy:   {val_acc:.4f}\n")
        f.write(f"Test Accuracy:  {test_acc:.4f}\n\n")
        f.write(f"Train Demographic Parity Difference: {train_dp:.4f}\n")
        f.write(f"Val Demographic Parity Difference:   {val_dp:.4f}\n")
        f.write(f"Test Demographic Parity Difference:  {test_dp:.4f}\n\n")
        f.write(f"Train Equalized Odds Difference: {train_eod:.4f}\n")
        f.write(f"Val Equalized Odds Difference:   {val_eod:.4f}\n")
        f.write(f"Test Equalized Odds Difference:  {test_eod:.4f}\n")

    print(f"\nSaved results to: {results_path}")


if __name__ == "__main__":
    main()
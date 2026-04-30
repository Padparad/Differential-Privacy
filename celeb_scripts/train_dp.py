from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from opacus import PrivacyEngine


def load_tsv_dataset(path: Path):
    data = np.loadtxt(path, delimiter="\t")
    y = data[:, 0].astype(np.float32)
    X = data[:, 1:].astype(np.float32)
    return X, y


def load_sensitive(path: Path):
    return np.loadtxt(path, delimiter=",").astype(int)


def demographic_parity_difference(y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    group0 = y_pred[sensitive == 0]
    group1 = y_pred[sensitive == 1]

    rate0 = np.mean(group0 == 1)
    rate1 = np.mean(group1 == 1)

    return abs(rate1 - rate0)


def equalized_odds_difference(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    group0_pos = (sensitive == 0) & (y_true == 1)
    group1_pos = (sensitive == 1) & (y_true == 1)

    group0_neg = (sensitive == 0) & (y_true == 0)
    group1_neg = (sensitive == 1) & (y_true == 0)

    def safe_mean(arr):
        if len(arr) == 0:
            return 0.0
        return np.mean(arr)

    group0_tpr = safe_mean(y_pred[group0_pos] == 1)
    group1_tpr = safe_mean(y_pred[group1_pos] == 1)

    group0_fpr = safe_mean(y_pred[group0_neg] == 1)
    group1_fpr = safe_mean(y_pred[group1_neg] == 1)

    return max(abs(group1_tpr - group0_tpr), abs(group1_fpr - group0_fpr))


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).cpu().numpy().astype(int)
    return preds


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "celeba_outputs"

    X_train, y_train = load_tsv_dataset(output_dir / "train_processed.tsv")
    X_val, y_val = load_tsv_dataset(output_dir / "val_processed.tsv")
    X_test, y_test = load_tsv_dataset(output_dir / "test_processed.tsv")

    a_train = load_sensitive(output_dir / "train_sensitive.csv")
    a_val = load_sensitive(output_dir / "val_sensitive.csv")
    a_test = load_sensitive(output_dir / "test_sensitive.csv")

    print(f"Train shape: {X_train.shape}")
    print(f"Val shape:   {X_val.shape}")
    print(f"Test shape:  {X_test.shape}")

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        drop_last=False
    )

    input_dim = X_train.shape[1]
    model = LogisticRegressionModel(input_dim)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=1.0,
        target_delta=1e-5,
        epochs=15,
        max_grad_norm=1.0,
    )

    num_epochs = 15

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1:02d} | Loss: {avg_loss:.4f} | ε: {epsilon:.4f}")

    y_train_pred = evaluate_model(model, X_train, y_train)
    y_val_pred = evaluate_model(model, X_val, y_val)
    y_test_pred = evaluate_model(model, X_test, y_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_dp = demographic_parity_difference(y_train_pred, a_train)
    val_dp = demographic_parity_difference(y_val_pred, a_val)
    test_dp = demographic_parity_difference(y_test_pred, a_test)

    train_eod = equalized_odds_difference(y_train.astype(int), y_train_pred, a_train)
    val_eod = equalized_odds_difference(y_val.astype(int), y_val_pred, a_val)
    test_eod = equalized_odds_difference(y_test.astype(int), y_test_pred, a_test)

    final_epsilon = privacy_engine.get_epsilon(delta=1e-5)

    print("\n=== DP Logistic Regression Results ===")
    print(f"Final epsilon: {final_epsilon:.4f}")
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

    results_path = output_dir / "dp_results_epsilon_50.txt"
    with open(results_path, "w") as f:
        f.write("=== DP Logistic Regression Results ===\n")
        f.write(f"Final epsilon: {final_epsilon:.4f}\n")
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
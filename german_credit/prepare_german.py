from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# Original German column names mapped to English (from codetable.txt)
COLUMN_NAMES = [
    "status",               # laufkont  – checking account status (ordinal 1–4)
    "duration",             # laufzeit  – credit duration in months (numeric)
    "credit_history",       # moral     – credit history (ordinal 0–4)
    "purpose",              # verw      – purpose of credit (nominal 0–10)
    "amount",               # hoehe     – credit amount (numeric)
    "savings",              # sparkont  – savings account (ordinal 1–5)
    "employment_duration",  # beszeit   – employment duration (ordinal 1–5)
    "installment_rate",     # rate      – installment rate % of income (ordinal 1–4)
    "personal_status_sex",  # famges    – personal status & sex (nominal 1–4)
    "other_debtors",        # buerge    – other debtors/guarantors (nominal 1–3)
    "present_residence",    # wohnzeit  – present residence duration (ordinal 1–4)
    "property",             # verm      – most valuable property (ordinal 1–4)
    "age",                  # alter     – age in years (numeric)
    "other_installment_plans",  # weitkred – other installment plans (nominal 1–3)
    "housing",              # wohn      – housing (nominal 1–3)
    "number_credits",       # bishkred  – number of existing credits (ordinal 1–4)
    "job",                  # beruf     – job type (ordinal 1–4)
    "people_liable",        # pers      – number of people liable (binary 1–2)
    "telephone",            # telef     – telephone (binary 1–2)
    "foreign_worker",       # gastarb   – foreign worker (binary 1–2)
    "credit_risk",          # kredit    – TARGET: 0=bad, 1=good
]

# Truly continuous numeric features
NUMERIC_FEATURES = [
    "duration",
    "amount",
    "age",
]

# Ordinal features — kept as integers and scaled (order is meaningful)
ORDINAL_FEATURES = [
    "status",
    "credit_history",
    "savings",
    "employment_duration",
    "installment_rate",
    "present_residence",
    "property",
    "number_credits",
    "job",
]

# Nominal categorical features — one-hot encoded
CATEGORICAL_FEATURES = [
    "purpose",
    "personal_status_sex",
    "other_debtors",
    "other_installment_plans",
    "housing",
    "people_liable",
    "telephone",
    "foreign_worker",
]

# Sensitive attribute: sex, derived from personal_status_sex
# Encoding: 0 = female (codes 2 and 4), 1 = male (codes 1 and 3)
def encode_sensitive_attribute(df: pd.DataFrame) -> np.ndarray:
    pss = df["personal_status_sex"]
    # 1: male divorced/sep, 3: male married/widowed → male=1
    # 2: female non-single OR male single, 4: female single → female=0
    # Code 2 is ambiguous but conventionally treated as female in this dataset.
    is_male = pss.isin([1, 3]).astype(int)
    return is_male.to_numpy()


def encode_label(df: pd.DataFrame) -> np.ndarray:
    # 0 = bad credit risk, 1 = good credit risk
    return df["credit_risk"].to_numpy().astype(int)


def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    ordinal_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("ord", ordinal_transformer, ORDINAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def save_processed_split(output_path: Path, X: np.ndarray, y: np.ndarray):
    arr = np.column_stack([y, X])
    np.savetxt(output_path, arr, delimiter="\t", fmt="%.6f")


def save_sensitive_split(output_path: Path, a: np.ndarray):
    np.savetxt(output_path, a, delimiter=",", fmt="%d")


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "south_german_credit"
    output_dir = project_root / "credit_outputs"
    output_dir.mkdir(exist_ok=True)

    data_path = data_dir / "SouthGermanCredit.asc"

    df = pd.read_csv(
        data_path,
        sep=" ",
        header=0,
        names=COLUMN_NAMES,
        skiprows=1,   # skip the original German header line
    )

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Credit risk distribution:\n{df['credit_risk'].value_counts()}\n")

    # Labels and sensitive attribute (before any dropping)
    y_all = encode_label(df)
    a_all = encode_sensitive_attribute(df)

    # Features: drop the label; keep personal_status_sex for now so the
    # categorical encoder sees it, but sex signal is captured in a_all.
    # We drop it from model input to avoid directly encoding sex.
    drop_columns = ["credit_risk", "personal_status_sex"]
    X_df = df.drop(columns=drop_columns)

    # Update feature lists to reflect dropped column
    cat_features_model = [f for f in CATEGORICAL_FEATURES if f != "personal_status_sex"]

    def build_preprocessor_model():
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        ordinal_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, NUMERIC_FEATURES),
                ("ord", ordinal_transformer, ORDINAL_FEATURES),
                ("cat", categorical_transformer, cat_features_model),
            ]
        )
        return preprocessor

    # Train / val / test split: 60% / 20% / 20%
    indices = np.arange(len(df))
    idx_trainval, idx_test = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=y_all
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.25, random_state=42, stratify=y_all[idx_trainval]
    )  # 0.25 * 0.80 = 0.20 of total

    X_train_df = X_df.iloc[idx_train].reset_index(drop=True)
    X_val_df   = X_df.iloc[idx_val].reset_index(drop=True)
    X_test_df  = X_df.iloc[idx_test].reset_index(drop=True)

    y_train, y_val, y_test = y_all[idx_train], y_all[idx_val], y_all[idx_test]
    a_train, a_val, a_test = a_all[idx_train], a_all[idx_val], a_all[idx_test]

    preprocessor = build_preprocessor_model()
    X_train = preprocessor.fit_transform(X_train_df)
    X_val   = preprocessor.transform(X_val_df)
    X_test  = preprocessor.transform(X_test_df)

    # Sanity checks
    for name, X in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        assert not np.isnan(X).any(), f"{name} contains NaN"
        assert not np.isinf(X).any(), f"{name} contains Inf"
        print(f"{name}: shape={X.shape}")

    print(f"\ny_train distribution: {np.bincount(y_train)}")
    print(f"y_val   distribution: {np.bincount(y_val)}")
    print(f"y_test  distribution: {np.bincount(y_test)}")
    print(f"\na_train (0=female, 1=male): {np.bincount(a_train)}")
    print(f"a_val   (0=female, 1=male): {np.bincount(a_val)}")
    print(f"a_test  (0=female, 1=male): {np.bincount(a_test)}")

    # Save
    save_processed_split(output_dir / "train_processed.tsv", X_train, y_train)
    save_processed_split(output_dir / "val_processed.tsv",   X_val,   y_val)
    save_processed_split(output_dir / "test_processed.tsv",  X_test,  y_test)

    save_sensitive_split(output_dir / "train_sensitive.csv", a_train)
    save_sensitive_split(output_dir / "val_sensitive.csv",   a_val)
    save_sensitive_split(output_dir / "test_sensitive.csv",  a_test)

    feature_names = preprocessor.get_feature_names_out()
    pd.Series(feature_names).to_csv(output_dir / "feature_names.csv", index=False)

    print("\nPreprocessing complete. Saved files to:", output_dir)


if __name__ == "__main__":
    main()
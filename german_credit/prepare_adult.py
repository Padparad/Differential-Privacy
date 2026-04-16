from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]


def load_adult_train(train_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        train_path,
        header=None,
        names=COLUMN_NAMES,
        na_values="?",
        skipinitialspace=True
    )
    return df


def load_adult_test(test_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        test_path,
        header=None,
        names=COLUMN_NAMES,
        na_values="?",
        skipinitialspace=True,
        skiprows=1  # adult.test has a first line that should be skipped
    )
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strip spaces from string columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()

    # Remove trailing period in adult.test labels, e.g. ">50K."
    df["income"] = df["income"].str.replace(".", "", regex=False)

    # Drop rows with missing values for now (simple and standard for first version)
    df = df.dropna().reset_index(drop=True)

    return df


def encode_label(df: pd.DataFrame) -> pd.Series:
    return df["income"].map({"<=50K": 0, ">50K": 1}).astype(int)


def encode_sensitive_attribute(df: pd.DataFrame) -> pd.Series:
    # Female = 0, Male = 1
    return df["sex"].map({"Female": 0, "Male": 1}).astype(int)


def build_preprocessor():
    numeric_features = [
        "age",
        "fnlwgt",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week"
    ]

    categorical_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "native_country"
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, numeric_features, categorical_features


def save_processed_split(output_path: Path, X: np.ndarray, y: np.ndarray):
    arr = np.column_stack([y, X])
    np.savetxt(output_path, arr, delimiter="\t", fmt="%.6f")


def save_sensitive_split(output_path: Path, a: np.ndarray):
    np.savetxt(output_path, a, delimiter=",", fmt="%d")


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"

    # Load raw files
    train_df = load_adult_train(train_path)
    test_df = load_adult_test(test_path)

    # Clean
    train_df = clean_dataframe(train_df)
    test_df = clean_dataframe(test_df)

    # Split train into train/val
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=encode_label(train_df)
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Labels
    y_train = encode_label(train_df).to_numpy()
    y_val = encode_label(val_df).to_numpy()
    y_test = encode_label(test_df).to_numpy()

    # Sensitive attribute
    a_train = encode_sensitive_attribute(train_df).to_numpy()
    a_val = encode_sensitive_attribute(val_df).to_numpy()
    a_test = encode_sensitive_attribute(test_df).to_numpy()

    # Features: remove label and remove sensitive attribute from model input
    drop_columns = ["income", "sex"]
    X_train_df = train_df.drop(columns=drop_columns)
    X_val_df = val_df.drop(columns=drop_columns)
    X_test_df = test_df.drop(columns=drop_columns)

    # Preprocess
    preprocessor, _, _ = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    # Save processed files
    save_processed_split(output_dir / "train_processed.tsv", X_train, y_train)
    save_processed_split(output_dir / "val_processed.tsv", X_val, y_val)
    save_processed_split(output_dir / "test_processed.tsv", X_test, y_test)

    save_sensitive_split(output_dir / "train_sensitive.csv", a_train)
    save_sensitive_split(output_dir / "val_sensitive.csv", a_val)
    save_sensitive_split(output_dir / "test_sensitive.csv", a_test)

    # Save feature names for debugging / later interpretation
    feature_names = preprocessor.get_feature_names_out()
    pd.Series(feature_names).to_csv(output_dir / "feature_names.csv", index=False)

    print("Preprocessing complete.")
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape:   {X_val.shape}")
    print(f"Test shape:  {X_test.shape}")
    print("Saved files:")
    print(output_dir / "train_processed.tsv")
    print(output_dir / "val_processed.tsv")
    print(output_dir / "test_processed.tsv")
    print(output_dir / "train_sensitive.csv")
    print(output_dir / "val_sensitive.csv")
    print(output_dir / "test_sensitive.csv")
    print(output_dir / "feature_names.csv")


if __name__ == "__main__":
    main()
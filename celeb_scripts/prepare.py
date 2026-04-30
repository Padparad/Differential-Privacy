from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# CelebA file reference:
#   list_attr_celeba.csv        - 40 binary attributes per image (+1/-1 encoding)
#   list_eval_partition.csv     - Official train/val/test split (0/1/2)
#   list_landmarks_align_celeba.csv - 5 facial landmark (x,y) coordinates
#   list_bbox_celeba.csv        - Face bounding box (x_1, y_1, width, height)
#
# Label:
#   has_beard = 1 if No_Beard == -1, else 0  (~16.5% positive rate)
#   Note: we do NOT use Goatee / Mustache / 5_o_Clock_Shadow / Sideburns as
#   features — they would directly leak the label.
#
# Sensitive attribute:
#   Young: remapped from {-1, 1} → {0=Not Young, 1=Young}  (~77% Young)
#
# Features:
#   Landmark coordinates (lefteye_x/y, righteye_x/y, nose_x/y,
#                         leftmouth_x/y, rightmouth_x/y) — 10 values
#   Bounding box (x_1, y_1, width, height) — 4 values
#   Other CelebA attributes that do NOT leak beard information — 29 values
#   (All beard-related and gender columns are excluded from features)
#
# Split: official CelebA partition (0=train, 1=val, 2=test)


# Attributes excluded from features:
#   - Label-leaking beard attributes: No_Beard, 5_o_Clock_Shadow, Goatee,
#     Mustache, Sideburns
#   - Sensitive attribute used separately: Young
EXCLUDE_ATTRS = {
    "No_Beard",         # used to construct the label
    "5_o_Clock_Shadow", # strongly correlated with beard
    "Goatee",           # beard subtype
    "Mustache",         # beard subtype
    "Sideburns",        # beard subtype
    "Young",            # used as sensitive attribute
}

LANDMARK_COLS = [
    "lefteye_x", "lefteye_y",
    "righteye_x", "righteye_y",
    "nose_x", "nose_y",
    "leftmouth_x", "leftmouth_y",
    "rightmouth_x", "rightmouth_y",
]

BBOX_COLS = ["x_1", "y_1", "width", "height"]


def load_data(data_dir: Path) -> pd.DataFrame:
    attr = pd.read_csv(data_dir / "list_attr_celeba.csv")
    part = pd.read_csv(data_dir / "list_eval_partition.csv")
    lm   = pd.read_csv(data_dir / "list_landmarks_align_celeba.csv")
    bbox = pd.read_csv(data_dir / "list_bbox_celeba.csv")

    df = attr.merge(part, on="image_id") \
             .merge(lm,   on="image_id") \
             .merge(bbox,  on="image_id")

    return df


def encode_label(df: pd.DataFrame) -> np.ndarray:
    # No_Beard == -1  →  has beard = 1
    # No_Beard ==  1  →  no beard  = 0
    return (df["No_Beard"] == -1).astype(int).to_numpy()


def encode_sensitive_attribute(df: pd.DataFrame) -> np.ndarray:
    # Young: 1 → 1, -1 (Not Young) → 0
    return df["Young"].map({1: 1, -1: 0}).astype(int).to_numpy()


def get_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    # CelebA attribute columns (excluding image_id, label-leaking, and sensitive)
    all_attr_cols = [
        c for c in df.columns
        if c not in {"image_id", "partition"} and c not in EXCLUDE_ATTRS
        and c not in LANDMARK_COLS and c not in BBOX_COLS
    ]
    # Remap remaining attributes from {-1, 1} → {0, 1}
    attr_df = df[all_attr_cols].replace({-1: 0, 1: 1})

    lm_df   = df[LANDMARK_COLS].astype(np.float32)
    bbox_df = df[BBOX_COLS].astype(np.float32)

    return pd.concat([lm_df, bbox_df, attr_df], axis=1)


def build_preprocessor(numeric_cols, binary_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Binary {0,1} attributes are already on a consistent scale;
    # still scale them so all features live in the same range
    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("bin", binary_transformer, binary_cols),
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
    data_dir   = project_root / "archive"
    output_dir = project_root / "celeba_outputs"
    output_dir.mkdir(exist_ok=True)

    print("Loading data...")
    df = load_data(data_dir)
    print(f"Total samples: {len(df)}")

    # Use the official CelebA partition
    train_df = df[df["partition"] == 0].reset_index(drop=True)
    val_df   = df[df["partition"] == 1].reset_index(drop=True)
    test_df  = df[df["partition"] == 2].reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Labels
    y_train = encode_label(train_df)
    y_val   = encode_label(val_df)
    y_test  = encode_label(test_df)

    print(f"Beard prevalence — Train: {y_train.mean():.4f} | Val: {y_val.mean():.4f} | Test: {y_test.mean():.4f}")

    # Sensitive attribute
    a_train = encode_sensitive_attribute(train_df)
    a_val   = encode_sensitive_attribute(val_df)
    a_test  = encode_sensitive_attribute(test_df)

    # Feature DataFrames
    X_train_df = get_feature_df(train_df)
    X_val_df   = get_feature_df(val_df)
    X_test_df  = get_feature_df(test_df)

    # Identify numeric vs binary columns
    numeric_cols = LANDMARK_COLS + BBOX_COLS
    binary_cols  = [c for c in X_train_df.columns if c not in numeric_cols]

    print(f"Numeric features:  {len(numeric_cols)}")
    print(f"Binary features:   {len(binary_cols)}")
    print(f"Total features:    {len(numeric_cols) + len(binary_cols)}")

    # Preprocess (fit only on train)
    preprocessor = build_preprocessor(numeric_cols, binary_cols)
    X_train = preprocessor.fit_transform(X_train_df)
    X_val   = preprocessor.transform(X_val_df)
    X_test  = preprocessor.transform(X_test_df)

    # Save processed files
    save_processed_split(output_dir / "train_processed.tsv", X_train, y_train)
    save_processed_split(output_dir / "val_processed.tsv",   X_val,   y_val)
    save_processed_split(output_dir / "test_processed.tsv",  X_test,  y_test)

    save_sensitive_split(output_dir / "train_sensitive.csv", a_train)
    save_sensitive_split(output_dir / "val_sensitive.csv",   a_val)
    save_sensitive_split(output_dir / "test_sensitive.csv",  a_test)

    # Save feature names
    feature_names = preprocessor.get_feature_names_out()
    pd.Series(feature_names).to_csv(output_dir / "feature_names.csv", index=False)

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Val shape:   {X_val.shape}")
    print(f"Test shape:  {X_test.shape}")
    print("\nSaved files:")
    for fname in [
        "train_processed.tsv", "val_processed.tsv", "test_processed.tsv",
        "train_sensitive.csv", "val_sensitive.csv", "test_sensitive.csv",
        "feature_names.csv",
    ]:
        print(f"  {output_dir / fname}")


if __name__ == "__main__":
    main()
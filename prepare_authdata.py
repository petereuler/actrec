import argparse
import csv
from pathlib import Path

import pandas as pd


FILE_PATTERNS = {
    "data_time_domain.txt": "*_mobile_data_time_domain_resample.csv",
    "data_frequency_domain.txt": "*_mobile_data_frequency_domain.csv",
    "data_feature_time_domain.txt": "*_mobile_data_feature_time_domain.csv",
    "data_feature_frequency_domain.txt": "*_mobile_data_feature_frequency_domain.csv",
}

BASE_TABLE = "data_time_domain.txt"


def _read_table(path: Path, sep: str | None = None) -> pd.DataFrame:
    if sep is not None:
        return pd.read_csv(path, sep=sep)
    return pd.read_csv(path, sep=None, engine="python")


def _detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except Exception:
        return "\t"


def _load_template_columns(template_root: Path) -> dict[str, list[str]]:
    template_columns: dict[str, list[str]] = {}
    for output_name in FILE_PATTERNS:
        template_file = template_root / output_name
        if not template_file.exists():
            continue
        template_sep = _detect_delimiter(template_file)
        template_columns[output_name] = _read_table(template_file, sep=template_sep).columns.tolist()
    return template_columns


def _collect_nonempty_files(root: Path, pattern: str) -> list[Path]:
    files = [p for p in root.rglob(pattern) if p.is_file() and p.stat().st_size > 0]
    files.sort()
    return files


def _extract_user_from_path(file_path: Path) -> str | None:
    for part in file_path.parts:
        if part.startswith("GestureData_Sample_"):
            pieces = part.split("_")
            if len(pieces) >= 4:
                return pieces[2]
    return None


def _count_candidate_files(root: Path, pattern: str) -> tuple[int, int, dict[str, int]]:
    total = 0
    empty = 0
    empty_by_user: dict[str, int] = {}
    for file_path in root.rglob(pattern):
        if not file_path.is_file():
            continue
        total += 1
        if file_path.stat().st_size == 0:
            empty += 1
            user = _extract_user_from_path(file_path)
            if user is not None:
                empty_by_user[user] = empty_by_user.get(user, 0) + 1
    return total, empty, empty_by_user


def _concat_files(
    files: list[Path], expected_columns: list[str] | None = None
) -> tuple[pd.DataFrame, int]:
    frames: list[pd.DataFrame] = []
    read_error_count = 0
    for file_path in files:
        try:
            df = _read_table(file_path)
        except Exception:
            read_error_count += 1
            continue
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=expected_columns if expected_columns else None), read_error_count

    merged = pd.concat(frames, ignore_index=True)
    if expected_columns:
        # 对齐到 data 目录已有文件的列顺序，保证格式完全一致。
        for col in expected_columns:
            if col not in merged.columns:
                merged[col] = pd.NA
        merged = merged[expected_columns]
    return merged, read_error_count


def _format_top_counts(counts: dict[str, int], limit: int = 10) -> str:
    if not counts:
        return "none"
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return ", ".join(f"{user}:{count}" for user, count in items)


def build_auth_dataset(auth_root: Path, output_root: Path, template_root: Path | None) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    template_columns = _load_template_columns(template_root) if template_root else {}

    for output_name, pattern in FILE_PATTERNS.items():
        total_candidates, empty_candidates, empty_by_user = _count_candidate_files(
            auth_root, pattern
        )
        files = _collect_nonempty_files(auth_root, pattern)
        expected_columns = template_columns.get(output_name)
        merged, read_error_count = _concat_files(files, expected_columns=expected_columns)
        out_file = output_root / output_name
        sep = "\t"
        if template_root:
            template_file = template_root / output_name
            if template_file.exists():
                sep = _detect_delimiter(template_file)
        merged.to_csv(out_file, index=False, sep=sep)
        print(
            f"[OK] {output_name}: candidates={total_candidates} "
            f"nonempty={len(files)} empty={empty_candidates} read_error={read_error_count} "
            f"-> {len(merged)} rows"
        )
        if empty_candidates > 0:
            print(
                f"[WARN] {output_name}: empty-file users(top)={_format_top_counts(empty_by_user)}"
            )


def _split_file_ids_by_user(
    base_df: pd.DataFrame, train_ratio: float, random_seed: int
) -> tuple[set[str], set[str]]:
    if "user" not in base_df.columns or "file_id" not in base_df.columns:
        raise ValueError("Base table must contain 'user' and 'file_id' columns.")

    train_ids: set[str] = set()
    test_ids: set[str] = set()

    for user, grp in base_df.groupby("user", sort=False):
        _ = user  # keep readable loop variable for debug extensions
        file_ids = grp["file_id"].astype(str).drop_duplicates().sample(
            frac=1.0, random_state=random_seed
        ).tolist()
        n = len(file_ids)
        if n == 0:
            continue

        n_train = int(round(n * train_ratio))
        if n > 1:
            n_train = min(max(n_train, 1), n - 1)
        else:
            n_train = 1

        train_ids.update(file_ids[:n_train])
        test_ids.update(file_ids[n_train:])

    return train_ids, test_ids


def split_auth_dataset_by_user(
    merged_root: Path,
    out_root: Path,
    train_ratio: float = 0.6,
    random_seed: int = 42,
) -> None:
    out_train = out_root / "train"
    out_test = out_root / "test"
    out_train.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)

    base_file = merged_root / BASE_TABLE
    if not base_file.exists():
        raise FileNotFoundError(f"Base table not found: {base_file}")

    sep = _detect_delimiter(base_file)
    base_df = _read_table(base_file, sep=sep)
    train_ids, test_ids = _split_file_ids_by_user(
        base_df=base_df, train_ratio=train_ratio, random_seed=random_seed
    )

    if train_ids & test_ids:
        raise RuntimeError("Split error: overlapping file_id between train and test.")

    for output_name in FILE_PATTERNS:
        file_path = merged_root / output_name
        if not file_path.exists():
            continue
        df = _read_table(file_path, sep=sep)
        if "file_id" not in df.columns:
            raise ValueError(f"'file_id' column missing in {file_path}")
        file_ids = df["file_id"].astype(str)
        train_df = df[file_ids.isin(train_ids)].copy()
        test_df = df[file_ids.isin(test_ids)].copy()

        train_df.to_csv(out_train / output_name, index=False, sep=sep)
        test_df.to_csv(out_test / output_name, index=False, sep=sep)
        print(
            f"[SPLIT] {output_name}: train={len(train_df)} rows, test={len(test_df)} rows"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert scattered authdata CSV files into the unified data/*.txt-style tables."
    )
    parser.add_argument("--auth-root", default="authdata", help="Root directory of raw authdata")
    parser.add_argument("--out-root", default="data/auth", help="Output directory")
    parser.add_argument(
        "--template-root",
        default="data/train",
        help="Directory that contains data_time_domain.txt etc. for column-order template",
    )
    parser.add_argument(
        "--split-by-user",
        action="store_true",
        help="Split merged auth dataset into train/test by each user's samples",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Train ratio for per-user split (test ratio is 1-train_ratio)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for per-user split",
    )
    args = parser.parse_args()

    auth_root = Path(args.auth_root)
    out_root = Path(args.out_root)
    template_root = Path(args.template_root) if args.template_root else None

    if not auth_root.exists():
        raise FileNotFoundError(f"auth root not found: {auth_root}")

    build_auth_dataset(auth_root=auth_root, output_root=out_root, template_root=template_root)

    if args.split_by_user:
        if not (0.0 < args.train_ratio < 1.0):
            raise ValueError("--train-ratio must be in (0, 1).")
        split_auth_dataset_by_user(
            merged_root=out_root,
            out_root=out_root,
            train_ratio=args.train_ratio,
            random_seed=args.split_seed,
        )


if __name__ == "__main__":
    main()

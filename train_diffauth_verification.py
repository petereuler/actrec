from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from DiffAuth import DiffAuthModel, LearnableResidualLoss, create_pair_indices
from data_loader import load_time_series_with_meta


ACTION_TO_SUBLABELS = {
    "shakeA": ("10201", "10202", "10203"),
    "shakeB": ("10301",),
}
RANDOM_SUBLABEL = "999"
SCENARIO_ORDER = (
    "target_enrolled",
    "same_user_other_action",
    "same_user_random",
    "other_user_same_action",
    "other_user_other_action",
    "other_user_random",
)
SCENARIO_WEIGHTS = {
    "target_enrolled": 0.40,
    "same_user_other_action": 0.25,
    "same_user_random": 0.15,
    "other_user_same_action": 0.10,
    "other_user_other_action": 0.05,
    "other_user_random": 0.05,
}


@dataclass
class RawSamples:
    x: np.ndarray
    users: np.ndarray
    sub_labels: np.ndarray
    file_ids: np.ndarray
    source_files: np.ndarray

    def subset(self, mask: np.ndarray) -> "RawSamples":
        return RawSamples(
            x=self.x[mask],
            users=self.users[mask],
            sub_labels=self.sub_labels[mask],
            file_ids=self.file_ids[mask],
            source_files=self.source_files[mask],
        )


@dataclass
class VerificationSplit:
    x: np.ndarray
    y: np.ndarray
    users: np.ndarray
    sub_labels: np.ndarray
    file_ids: np.ndarray
    scenarios: np.ndarray


@dataclass
class UserProblem:
    train_split: VerificationSplit
    enroll_split: VerificationSplit
    val_split: VerificationSplit
    test_split: VerificationSplit
    train_rows: int
    test_rows: int


@dataclass
class ThresholdSelection:
    threshold: float
    eer: float
    eer_threshold: float
    far: float
    frr: float
    hter: float
    mode: str


class VerifyDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class SimpleTimeBackbone(nn.Module):
    """Lightweight 1D CNN backbone for IMU time series."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, feat_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_raw_from_files(file_paths: list[str]) -> RawSamples:
    xs: list[np.ndarray] = []
    users_list: list[np.ndarray] = []
    sub_labels_list: list[np.ndarray] = []
    file_ids_list: list[np.ndarray] = []
    source_files_list: list[np.ndarray] = []

    for fp in file_paths:
        x, meta = load_time_series_with_meta(fp)
        prefix = Path(fp).stem
        file_ids = np.array([f"{prefix}:{fid}" for fid in meta["file_ids"]], dtype=str)

        xs.append(x)
        users_list.append(meta["users"].astype(str))
        sub_labels_list.append(meta["sub_labels"].astype(str))
        file_ids_list.append(file_ids)
        source_files_list.append(np.array([str(fp)] * len(x), dtype=str))

    return RawSamples(
        x=np.concatenate(xs, axis=0),
        users=np.concatenate(users_list, axis=0),
        sub_labels=np.concatenate(sub_labels_list, axis=0),
        file_ids=np.concatenate(file_ids_list, axis=0),
        source_files=np.concatenate(source_files_list, axis=0),
    )


def scenario_name(
    user: str,
    sub_label: str,
    target_user: str,
    enroll_sub_labels: set[str],
) -> str:
    if user == target_user and sub_label in enroll_sub_labels:
        return "target_enrolled"
    if user == target_user and sub_label == RANDOM_SUBLABEL:
        return "same_user_random"
    if user == target_user:
        return "same_user_other_action"
    if sub_label == RANDOM_SUBLABEL:
        return "other_user_random"
    if sub_label in enroll_sub_labels:
        return "other_user_same_action"
    return "other_user_other_action"


def build_split(
    raw: RawSamples,
    mask: np.ndarray,
    target_user: str,
    enroll_sub_labels: set[str],
) -> VerificationSplit:
    sub = raw.subset(mask)
    y = (
        (sub.users == str(target_user))
        & np.isin(sub.sub_labels, sorted(enroll_sub_labels))
    ).astype(np.int64)
    scenarios = np.array(
        [
            scenario_name(user, sub_label, str(target_user), enroll_sub_labels)
            for user, sub_label in zip(sub.users, sub.sub_labels)
        ],
        dtype=str,
    )
    return VerificationSplit(
        x=sub.x,
        y=y,
        users=sub.users,
        sub_labels=sub.sub_labels,
        file_ids=sub.file_ids,
        scenarios=scenarios,
    )


def choose_counts(n_items: int, val_ratio: float, enroll_ratio: float) -> tuple[int, int, int]:
    if n_items < 3:
        raise ValueError(
            f"Need at least 3 positive file groups for train/enroll/val, got {n_items}."
        )

    n_enroll = max(1, int(round(n_items * enroll_ratio)))
    n_val = max(1, int(round(n_items * val_ratio)))
    n_train = n_items - n_enroll - n_val

    while n_train < 1 and n_enroll > 1:
        n_enroll -= 1
        n_train += 1
    while n_train < 1 and n_val > 1:
        n_val -= 1
        n_train += 1
    if n_train < 1:
        raise ValueError(
            f"Unable to reserve train/enroll/val positive groups from {n_items} groups."
        )
    return n_train, n_enroll, n_val


def split_positive_ids(
    file_ids: np.ndarray,
    seed: int,
    val_ratio: float,
    enroll_ratio: float,
) -> tuple[set[str], set[str], set[str]]:
    unique_ids = list(dict.fromkeys(file_ids.tolist()))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    n_train, n_enroll, n_val = choose_counts(len(unique_ids), val_ratio, enroll_ratio)

    train_ids = set(unique_ids[:n_train])
    enroll_ids = set(unique_ids[n_train : n_train + n_enroll])
    val_ids = set(unique_ids[n_train + n_enroll : n_train + n_enroll + n_val])
    return train_ids, enroll_ids, val_ids


def split_train_val_ids(
    file_ids: np.ndarray,
    seed: int,
    val_ratio: float,
) -> tuple[set[str], set[str]]:
    unique_ids = list(dict.fromkeys(file_ids.tolist()))
    if len(unique_ids) < 2:
        raise ValueError(f"Need at least 2 file groups for train/val, got {len(unique_ids)}.")

    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    n_val = max(1, int(round(len(unique_ids) * val_ratio)))
    n_val = min(n_val, len(unique_ids) - 1)
    val_ids = set(unique_ids[:n_val])
    train_ids = set(unique_ids[n_val:])
    return train_ids, val_ids


def build_user_problem(
    train_raw: RawSamples,
    test_raw: RawSamples,
    target_user: str,
    enroll_action: str,
    val_ratio: float,
    enroll_ratio: float,
    seed: int,
) -> UserProblem:
    enroll_sub_labels = set(ACTION_TO_SUBLABELS[enroll_action])
    train_non_random_mask = train_raw.sub_labels != RANDOM_SUBLABEL

    train_positive_mask = train_non_random_mask & (
        (train_raw.users == str(target_user))
        & np.isin(train_raw.sub_labels, sorted(enroll_sub_labels))
    )
    train_negative_mask = train_non_random_mask & ~train_positive_mask

    train_pos_ids, enroll_ids, val_pos_ids = split_positive_ids(
        train_raw.file_ids[train_positive_mask],
        seed=seed,
        val_ratio=val_ratio,
        enroll_ratio=enroll_ratio,
    )
    train_neg_ids, val_neg_ids = split_train_val_ids(
        train_raw.file_ids[train_negative_mask],
        seed=seed + 17,
        val_ratio=val_ratio,
    )

    random_val_mask = train_raw.sub_labels == RANDOM_SUBLABEL

    train_mask = np.isin(train_raw.file_ids, sorted(train_pos_ids | train_neg_ids))
    enroll_mask = np.isin(train_raw.file_ids, sorted(enroll_ids))
    val_mask = np.isin(train_raw.file_ids, sorted(val_pos_ids | val_neg_ids)) | random_val_mask
    test_mask = np.ones(len(test_raw.x), dtype=bool)

    train_split = build_split(train_raw, train_mask, target_user, enroll_sub_labels)
    enroll_split = build_split(train_raw, enroll_mask, target_user, enroll_sub_labels)
    val_split = build_split(train_raw, val_mask, target_user, enroll_sub_labels)
    test_split = build_split(test_raw, test_mask, target_user, enroll_sub_labels)

    if train_split.y.sum() == 0 or enroll_split.y.sum() == 0 or val_split.y.sum() == 0:
        raise ValueError("Positive samples are missing after the train/enroll/val split.")
    if np.all(train_split.y == 1) or np.all(train_split.y == 0):
        raise ValueError("Training split must contain both positive and negative samples.")
    if np.all(val_split.y == 1) or np.all(val_split.y == 0):
        raise ValueError("Validation split must contain both positive and negative samples.")
    if np.all(test_split.y == 1) or np.all(test_split.y == 0):
        raise ValueError("Test split must contain both positive and negative samples.")

    return UserProblem(
        train_split=train_split,
        enroll_split=enroll_split,
        val_split=val_split,
        test_split=test_split,
        train_rows=len(train_split.y),
        test_rows=len(test_split.y),
    )


def make_loader(
    split: VerificationSplit,
    batch_size: int,
    num_workers: int,
    weighted: bool,
    seed: int,
) -> DataLoader:
    dataset = VerifyDataset(split.x, split.y)
    if not weighted:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_counts = np.bincount(split.y, minlength=2).astype(np.float32)
    class_weights = np.zeros_like(class_counts)
    nonzero_mask = class_counts > 0
    class_weights[nonzero_mask] = 1.0 / class_counts[nonzero_mask]
    sample_weights = class_weights[split.y]
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(split.y),
        replacement=True,
        generator=generator,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)


def template_consistency_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    neg_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pos_mask = labels == 1
    neg_mask = labels == 0
    zero = torch.zeros((), device=embeddings.device)

    if pos_mask.sum() == 0:
        return zero, {"pos_loss": zero, "neg_loss": zero}

    pos_embeddings = F.normalize(embeddings[pos_mask], dim=1)
    template = F.normalize(pos_embeddings.mean(dim=0, keepdim=True), dim=1)

    pos_cos = (pos_embeddings * template).sum(dim=1)
    pos_loss = 1.0 - pos_cos.mean()

    if neg_mask.sum() == 0:
        neg_loss = zero
    else:
        neg_embeddings = F.normalize(embeddings[neg_mask], dim=1)
        neg_cos = neg_embeddings @ template.squeeze(0)
        neg_loss = F.relu(neg_cos - neg_margin).mean()

    return pos_loss + neg_loss, {"pos_loss": pos_loss, "neg_loss": neg_loss}


def predict_split(
    model: DiffAuthModel,
    split: VerificationSplit,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        weighted=False,
        seed=0,
    )

    embeddings_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = model(x)
            embeddings = F.normalize(out["diff"], dim=1)
            probs = F.softmax(out["logits"], dim=1)[:, 1]
            embeddings_all.append(embeddings.cpu().numpy())
            probs_all.append(probs.cpu().numpy())

    return np.concatenate(embeddings_all, axis=0), np.concatenate(probs_all, axis=0)


def build_template(embeddings: np.ndarray) -> np.ndarray:
    template = embeddings.mean(axis=0)
    norm = np.linalg.norm(template) + 1e-12
    return template / norm


def fuse_scores(
    embeddings: np.ndarray,
    probs: np.ndarray,
    template: np.ndarray,
    template_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    cosine_scores = np.clip(embeddings @ template, -1.0, 1.0)
    template_scores = (cosine_scores + 1.0) * 0.5
    final_scores = template_weight * template_scores + (1.0 - template_weight) * probs
    return final_scores, template_scores


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return eer, threshold


def compute_binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(np.int64)

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    tp = int(np.sum(y_pred[pos_mask] == 1))
    fn = int(np.sum(y_pred[pos_mask] == 0))
    fp = int(np.sum(y_pred[neg_mask] == 1))
    tn = int(np.sum(y_pred[neg_mask] == 0))

    far = fp / max(int(neg_mask.sum()), 1)
    frr = fn / max(int(pos_mask.sum()), 1)
    tar = tp / max(int(pos_mask.sum()), 1)
    tnr = tn / max(int(neg_mask.sum()), 1)
    balanced_acc = 0.5 * (tar + tnr)

    eer, eer_threshold = compute_eer(y_true, scores)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "far": float(far),
        "frr": float(frr),
        "tar": float(tar),
        "tnr": float(tnr),
        "hter": float((far + frr) * 0.5),
        "balanced_acc": float(balanced_acc),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "threshold": float(threshold),
        "pos_rate": float(y_true.mean()),
    }


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_far: float,
) -> ThresholdSelection:
    thresholds = np.unique(np.concatenate(([0.0, 1.0], scores)))
    thresholds = np.sort(thresholds)[::-1]

    best_with_far: tuple[float, float, float] | None = None
    best_any: tuple[float, float, float] | None = None

    for thr in thresholds:
        metrics = compute_binary_metrics(y_true, scores, float(thr))
        far = metrics["far"]
        frr = metrics["frr"]
        hter = metrics["hter"]

        candidate = (frr, hter, float(thr))
        any_candidate = (abs(far - target_far) + hter, hter, float(thr))

        if far <= target_far + 1e-8:
            if best_with_far is None or candidate < best_with_far:
                best_with_far = candidate
        if best_any is None or any_candidate < best_any:
            best_any = any_candidate

    if best_with_far is not None:
        threshold = best_with_far[2]
        mode = "target_far"
    elif best_any is not None:
        threshold = best_any[2]
        mode = "closest_far"
    else:
        threshold = 0.5
        mode = "fallback"

    metrics = compute_binary_metrics(y_true, scores, threshold)
    return ThresholdSelection(
        threshold=float(threshold),
        eer=metrics["eer"],
        eer_threshold=metrics["eer_threshold"],
        far=metrics["far"],
        frr=metrics["frr"],
        hter=metrics["hter"],
        mode=mode,
    )


def compute_scenario_metrics(
    scenarios: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, float]]:
    decisions = (scores >= threshold).astype(np.int64)
    metrics: dict[str, dict[str, float]] = {}

    for scenario in SCENARIO_ORDER:
        mask = scenarios == scenario
        if not np.any(mask):
            continue
        accept_rate = float(decisions[mask].mean())
        metrics[scenario] = {
            "count": float(mask.sum()),
            "accept_rate": accept_rate,
            "mean_score": float(scores[mask].mean()),
        }
    return metrics


def scenario_risk(scenario_metrics: dict[str, dict[str, float]]) -> float:
    weighted_errors: list[float] = []
    weights: list[float] = []

    for scenario, weight in SCENARIO_WEIGHTS.items():
        if scenario not in scenario_metrics:
            continue
        accept_rate = scenario_metrics[scenario]["accept_rate"]
        error_rate = 1.0 - accept_rate if scenario == "target_enrolled" else accept_rate
        weighted_errors.append(weight * error_rate)
        weights.append(weight)

    if not weights:
        return 1.0
    return float(sum(weighted_errors) / sum(weights))


def summarize_split(name: str, split: VerificationSplit) -> None:
    pos = int(split.y.sum())
    neg = int(len(split.y) - pos)
    print(f"[{name}] rows={len(split.y)} pos={pos} neg={neg}")
    for scenario in SCENARIO_ORDER:
        count = int(np.sum(split.scenarios == scenario))
        if count > 0:
            print(f"[{name}] scenario={scenario} count={count}")


def format_scenario_metrics(scenario_metrics: dict[str, dict[str, float]]) -> str:
    parts: list[str] = []
    for scenario in SCENARIO_ORDER:
        if scenario not in scenario_metrics:
            continue
        info = scenario_metrics[scenario]
        parts.append(
            f"{scenario}:count={int(info['count'])},accept={info['accept_rate']:.4f},score={info['mean_score']:.4f}"
        )
    return " | ".join(parts)


def maybe_save_artifact(
    out_dir: str | None,
    target_user: str,
    enroll_action: str,
    model_state: dict[str, torch.Tensor],
    template: np.ndarray,
    threshold: float,
    args: argparse.Namespace,
) -> None:
    if not out_dir:
        return

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"verifier_user_{target_user}_{enroll_action}.pt"
    artifact = {
        "target_user": str(target_user),
        "enroll_action": enroll_action,
        "enroll_sub_labels": list(ACTION_TO_SUBLABELS[enroll_action]),
        "threshold": float(threshold),
        "template_weight": float(args.template_weight),
        "template": torch.from_numpy(template.astype(np.float32)),
        "model_state_dict": model_state,
        "model_config": {
            "feat_dim": args.feat_dim,
            "common_dim": args.common_dim,
            "diff_dim": args.diff_dim,
            "num_classes": 2,
        },
    }
    torch.save(artifact, artifact_path)

    metadata_path = artifact_path.with_suffix(".json")
    metadata = {
        "target_user": str(target_user),
        "enroll_action": enroll_action,
        "enroll_sub_labels": list(ACTION_TO_SUBLABELS[enroll_action]),
        "threshold": float(threshold),
        "template_weight": float(args.template_weight),
        "target_far": float(args.target_far),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[Artifact] saved={artifact_path}")


def train_epoch(
    model: DiffAuthModel,
    base_loss_fn: LearnableResidualLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    template_loss_weight: float,
    neg_margin: float,
) -> dict[str, float]:
    model.train()
    totals = {
        "total": 0.0,
        "base": 0.0,
        "template": 0.0,
        "samples": 0.0,
    }

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        pairs = create_pair_indices(batch_size=len(y), labels=y)
        base_loss, _ = base_loss_fn(
            output=out,
            backbone_feat=out["backbone_feat"],
            labels=y,
            pairs=pairs,
            classifier=model.classifier,
            common_classifier=model.common_classifier,
        )
        emb_loss, _ = template_consistency_loss(out["diff"], y, neg_margin=neg_margin)
        total_loss = base_loss + template_loss_weight * emb_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_size = len(y)
        totals["total"] += float(total_loss.item()) * batch_size
        totals["base"] += float(base_loss.item()) * batch_size
        totals["template"] += float(emb_loss.item()) * batch_size
        totals["samples"] += float(batch_size)

    denom = max(totals["samples"], 1.0)
    return {
        "loss": totals["total"] / denom,
        "base_loss": totals["base"] / denom,
        "template_loss": totals["template"] / denom,
    }


def evaluate_with_template(
    model: DiffAuthModel,
    enroll_split: VerificationSplit,
    eval_split: VerificationSplit,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    template_weight: float,
    threshold: float | None = None,
    target_far: float | None = None,
) -> dict[str, object]:
    enroll_embeddings, _ = predict_split(model, enroll_split, batch_size, num_workers, device)
    template = build_template(enroll_embeddings)

    eval_embeddings, eval_probs = predict_split(model, eval_split, batch_size, num_workers, device)
    fused_scores, template_scores = fuse_scores(
        embeddings=eval_embeddings,
        probs=eval_probs,
        template=template,
        template_weight=template_weight,
    )

    if threshold is None:
        if target_far is None:
            raise ValueError("target_far must be provided when threshold is not fixed.")
        selection = select_threshold(eval_split.y, fused_scores, target_far=target_far)
        threshold = selection.threshold
    else:
        metrics = compute_binary_metrics(eval_split.y, fused_scores, threshold)
        selection = ThresholdSelection(
            threshold=threshold,
            eer=metrics["eer"],
            eer_threshold=metrics["eer_threshold"],
            far=metrics["far"],
            frr=metrics["frr"],
            hter=metrics["hter"],
            mode="fixed",
        )

    binary_metrics = compute_binary_metrics(eval_split.y, fused_scores, threshold)
    scenarios = compute_scenario_metrics(eval_split.scenarios, fused_scores, threshold)
    return {
        "template": template,
        "scores": fused_scores,
        "template_scores": template_scores,
        "probs": eval_probs,
        "threshold_info": selection,
        "metrics": binary_metrics,
        "scenario_metrics": scenarios,
        "risk": scenario_risk(scenarios),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an action-aware biometric verifier for phone unlock."
    )
    parser.add_argument("--target-user", help="Target user id for one-vs-rest verification")
    parser.add_argument("--all-users", action="store_true", help="Run verification for all users")
    parser.add_argument(
        "--enroll-action",
        required=True,
        choices=sorted(ACTION_TO_SUBLABELS.keys()),
        help="Registered unlock gesture template to accept",
    )
    parser.add_argument(
        "--train-files",
        nargs="+",
        default=["data/auth/train/data_time_domain.txt", "data/reco/train/data_time_domain.txt"],
    )
    parser.add_argument(
        "--test-files",
        nargs="+",
        default=["data/auth/test/data_time_domain.txt", "data/reco/test/data_time_domain.txt"],
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--common-dim", type=int, default=64)
    parser.add_argument("--diff-dim", type=int, default=64)
    parser.add_argument("--template-weight", type=float, default=0.7)
    parser.add_argument("--template-loss-weight", type=float, default=0.25)
    parser.add_argument("--neg-margin", type=float, default=0.35)
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--enroll-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--artifact-dir",
        default="artifacts",
        help="Directory to save trained verifier artifacts",
    )
    return parser.parse_args()


def run_single_user(
    target_user: str,
    train_raw: RawSamples,
    test_raw: RawSamples,
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int = 0,
) -> dict[str, float] | None:
    set_seed(args.seed + seed_offset)

    try:
        problem = build_user_problem(
            train_raw=train_raw,
            test_raw=test_raw,
            target_user=target_user,
            enroll_action=args.enroll_action,
            val_ratio=args.val_ratio,
            enroll_ratio=args.enroll_ratio,
            seed=args.seed + seed_offset,
        )
    except ValueError as exc:
        print(f"[Skip] user={target_user} reason={exc}")
        return None

    print(f"\n[User {target_user}][Action {args.enroll_action}]")
    summarize_split("Train", problem.train_split)
    summarize_split("Enroll", problem.enroll_split)
    summarize_split("Val", problem.val_split)
    summarize_split("Test", problem.test_split)

    train_loader = make_loader(
        split=problem.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weighted=True,
        seed=args.seed + seed_offset,
    )

    backbone = SimpleTimeBackbone(feat_dim=args.feat_dim)
    model = DiffAuthModel(
        backbone=backbone,
        feat_dim=args.feat_dim,
        common_dim=args.common_dim,
        diff_dim=args.diff_dim,
        num_classes=2,
    ).to(device)

    base_loss_fn = LearnableResidualLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_risk = float("inf")
    best_threshold = 0.5
    best_template: np.ndarray | None = None
    best_val_metrics: dict[str, float] | None = None

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(
            model=model,
            base_loss_fn=base_loss_fn,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            template_loss_weight=args.template_loss_weight,
            neg_margin=args.neg_margin,
        )
        val_eval = evaluate_with_template(
            model=model,
            enroll_split=problem.enroll_split,
            eval_split=problem.val_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            template_weight=args.template_weight,
            target_far=args.target_far,
        )
        threshold_info: ThresholdSelection = val_eval["threshold_info"]  # type: ignore[assignment]
        val_metrics: dict[str, float] = val_eval["metrics"]  # type: ignore[assignment]
        risk = float(val_eval["risk"])

        if risk < best_val_risk:
            best_val_risk = risk
            best_threshold = threshold_info.threshold
            best_state = copy.deepcopy(model.state_dict())
            best_template = np.array(val_eval["template"], copy=True)
            best_val_metrics = val_metrics

        print(
            f"[User {target_user}][Epoch {epoch:03d}] "
            f"loss={train_stats['loss']:.4f} base={train_stats['base_loss']:.4f} "
            f"tpl={train_stats['template_loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} val_far={val_metrics['far']:.4f} "
            f"val_frr={val_metrics['frr']:.4f} val_hter={val_metrics['hter']:.4f} "
            f"val_risk={risk:.4f} thr={threshold_info.threshold:.4f} mode={threshold_info.mode}"
        )

    if best_state is None or best_template is None or best_val_metrics is None:
        return None

    model.load_state_dict(best_state)
    test_eval = evaluate_with_template(
        model=model,
        enroll_split=problem.enroll_split,
        eval_split=problem.test_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        template_weight=args.template_weight,
        threshold=best_threshold,
    )
    test_metrics: dict[str, float] = test_eval["metrics"]  # type: ignore[assignment]
    test_scenarios: dict[str, dict[str, float]] = test_eval["scenario_metrics"]  # type: ignore[assignment]

    print(
        f"[User {target_user}][Best] "
        f"val_hter={best_val_metrics['hter']:.4f} test_acc={test_metrics['acc']:.4f} "
        f"test_far={test_metrics['far']:.4f} test_frr={test_metrics['frr']:.4f} "
        f"test_eer={test_metrics['eer']:.4f} thr={best_threshold:.4f}"
    )
    print(f"[User {target_user}][Scenarios] {format_scenario_metrics(test_scenarios)}")

    maybe_save_artifact(
        out_dir=args.artifact_dir,
        target_user=target_user,
        enroll_action=args.enroll_action,
        model_state=best_state,
        template=best_template,
        threshold=best_threshold,
        args=args,
    )

    return {
        "user": target_user,
        "acc": test_metrics["acc"],
        "f1": test_metrics["f1"],
        "far": test_metrics["far"],
        "frr": test_metrics["frr"],
        "eer": test_metrics["eer"],
        "threshold": best_threshold,
        "risk": float(test_eval["risk"]),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if not (0.0 < args.val_ratio < 0.5):
        raise ValueError("--val-ratio must be between 0 and 0.5.")
    if not (0.0 < args.enroll_ratio < 0.5):
        raise ValueError("--enroll-ratio must be between 0 and 0.5.")
    if not (0.0 <= args.template_weight <= 1.0):
        raise ValueError("--template-weight must be in [0, 1].")
    if not (0.0 < args.target_far < 1.0):
        raise ValueError("--target-far must be in (0, 1).")

    train_raw = load_raw_from_files(args.train_files)
    test_raw = load_raw_from_files(args.test_files)

    candidate_users = sorted(
        set(train_raw.users.tolist()) | set(test_raw.users.tolist())
    )
    enroll_sub_labels = set(ACTION_TO_SUBLABELS[args.enroll_action])
    eligible_users = [
        user
        for user in candidate_users
        if np.any((train_raw.users == user) & np.isin(train_raw.sub_labels, sorted(enroll_sub_labels)))
        and np.any((test_raw.users == user) & np.isin(test_raw.sub_labels, sorted(enroll_sub_labels)))
    ]

    if args.all_users:
        target_users = eligible_users
    else:
        if not args.target_user:
            raise ValueError("Please provide --target-user or enable --all-users.")
        if str(args.target_user) not in eligible_users:
            raise ValueError(
                f"Target user {args.target_user} has no train/test samples for {args.enroll_action}."
            )
        target_users = [str(args.target_user)]

    print(
        f"[Setup] enroll_action={args.enroll_action} "
        f"train_rows={len(train_raw.x)} test_rows={len(test_raw.x)} "
        f"eligible_users={len(eligible_users)} selected={len(target_users)}"
    )
    print(f"[Setup] train_files={args.train_files}")
    print(f"[Setup] test_files={args.test_files}")
    print(f"[Setup] enroll_sub_labels={sorted(enroll_sub_labels)} random_excluded_from_train=True")

    results: list[dict[str, float]] = []
    for i, user in enumerate(target_users):
        result = run_single_user(
            target_user=user,
            train_raw=train_raw,
            test_raw=test_raw,
            args=args,
            device=device,
            seed_offset=i,
        )
        if result is not None:
            results.append(result)

    if not results:
        print("\n[Summary] No valid users were evaluated.")
        return

    print("\n[Summary By User]")
    for result in results:
        print(
            f"user={result['user']} acc={result['acc']:.4f} f1={result['f1']:.4f} "
            f"far={result['far']:.4f} frr={result['frr']:.4f} eer={result['eer']:.4f} "
            f"thr={result['threshold']:.4f} risk={result['risk']:.4f}"
        )

    mean_acc = float(np.mean([result["acc"] for result in results]))
    mean_f1 = float(np.mean([result["f1"] for result in results]))
    mean_far = float(np.mean([result["far"] for result in results]))
    mean_frr = float(np.mean([result["frr"] for result in results]))
    mean_eer = float(np.mean([result["eer"] for result in results]))
    mean_risk = float(np.mean([result["risk"] for result in results]))

    print("\n[Summary Mean]")
    print(
        f"users={len(results)} mean_acc={mean_acc:.4f} mean_f1={mean_f1:.4f} "
        f"mean_far={mean_far:.4f} mean_frr={mean_frr:.4f} mean_eer={mean_eer:.4f} "
        f"mean_risk={mean_risk:.4f}"
    )


if __name__ == "__main__":
    main()

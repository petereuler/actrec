from __future__ import annotations

import argparse
import copy
import json
import random
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from DiffAuth import DiffAuthModel, create_pair_indices
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
class JointTrainSplit:
    x: np.ndarray
    identity_labels: np.ndarray
    gate_labels: np.ndarray
    subaction_labels: np.ndarray
    users: np.ndarray
    sub_labels: np.ndarray
    file_ids: np.ndarray


@dataclass
class EvalSplit:
    x: np.ndarray
    users: np.ndarray
    sub_labels: np.ndarray
    file_ids: np.ndarray


@dataclass
class ThresholdSelection:
    threshold: float
    eer: float
    eer_threshold: float
    far: float
    frr: float
    hter: float
    mode: str


@dataclass
class JointProblem:
    train_split: JointTrainSplit
    enroll_split: EvalSplit
    val_split: EvalSplit
    test_split: EvalSplit
    eligible_users: list[str]
    user_to_index: dict[str, int]
    subaction_to_index: dict[str, int]


class JointTrainDataset(Dataset):
    def __init__(self, split: JointTrainSplit):
        self.x = torch.from_numpy(split.x).float()
        self.identity_labels = torch.from_numpy(split.identity_labels).long()
        self.gate_labels = torch.from_numpy(split.gate_labels).long()
        self.subaction_labels = torch.from_numpy(split.subaction_labels).long()

    def __len__(self) -> int:
        return len(self.gate_labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.identity_labels[idx], self.gate_labels[idx], self.subaction_labels[idx]


class EvalDataset(Dataset):
    def __init__(self, split: EvalSplit):
        self.x = torch.from_numpy(split.x).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return GradientReversalFn.apply(x, lambda_)


class TemporalResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.silu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.silu(x + residual)


class AttentiveStatsPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(x), dim=-1)
        mean = torch.sum(weights * x, dim=-1)
        var = torch.sum(weights * (x - mean.unsqueeze(-1)) ** 2, dim=-1)
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        return torch.cat([mean, std], dim=1)


class ArcMarginHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 24.0, margin: float = 0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale

        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-6))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale


class SimpleTimeBackbone(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            TemporalResBlock(32, 64, kernel_size=5, dilation=1),
            TemporalResBlock(64, 96, kernel_size=5, dilation=2),
            TemporalResBlock(96, 128, kernel_size=3, dilation=3),
        )
        self.pool = AttentiveStatsPool(128)
        self.fc = nn.Sequential(
            nn.Linear(256, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.fc(x)


class JointDiffAuthVerifier(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        common_dim: int,
        diff_dim: int,
        num_users: int,
        num_subactions: int,
    ):
        super().__init__()
        backbone = SimpleTimeBackbone(feat_dim=feat_dim)
        self.diffauth = DiffAuthModel(
            backbone=backbone,
            feat_dim=feat_dim,
            common_dim=common_dim,
            diff_dim=diff_dim,
            num_classes=num_users,
        )
        self.action_head = nn.Sequential(
            nn.Linear(common_dim, common_dim),
            nn.ReLU(),
            nn.Linear(common_dim, 2),
        )
        self.subaction_head = nn.Sequential(
            nn.Linear(common_dim, common_dim),
            nn.ReLU(),
            nn.Linear(common_dim, num_subactions),
        )
        self.common_identity_head = nn.Sequential(
            nn.Linear(common_dim, common_dim),
            nn.ReLU(),
            nn.Linear(common_dim, num_users),
        )
        self.common_to_diff = nn.Linear(common_dim, diff_dim)
        self.identity_margin_head = ArcMarginHead(diff_dim, num_users)
        self.use_arcface = False

    def forward(
        self,
        x: torch.Tensor,
        grl_lambda: float = 1.0,
        identity_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.diffauth(x)
        action_logits = self.action_head(out["common"])
        subaction_logits = self.subaction_head(out["common"])
        adv_common_logits = self.common_identity_head(grad_reverse(out["common"], grl_lambda))
        common_diff_proj = self.common_to_diff(out["common"])
        return {
            **out,
            "action_logits": action_logits,
            "subaction_logits": subaction_logits,
            "common_adv_logits": adv_common_logits,
            "common_diff_proj": common_diff_proj,
            "identity_margin_logits": (
                self.identity_margin_head(out["diff"], identity_labels)
                if self.use_arcface and identity_labels is not None
                else self.identity_margin_head(out["diff"])
            ),
        }


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


def unique_preserve_order(values: np.ndarray) -> list[str]:
    return list(dict.fromkeys(values.astype(str).tolist()))


def choose_counts(n_items: int, val_ratio: float, enroll_ratio: float) -> tuple[int, int, int]:
    if n_items < 3:
        raise ValueError(f"Need at least 3 file groups, got {n_items}.")

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
        raise ValueError(f"Unable to split {n_items} groups into train/enroll/val.")
    return n_train, n_enroll, n_val


def split_positive_ids(
    file_ids: np.ndarray,
    seed: int,
    val_ratio: float,
    enroll_ratio: float,
) -> tuple[set[str], set[str], set[str]]:
    unique_ids = unique_preserve_order(file_ids)
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    n_train, n_enroll, n_val = choose_counts(len(unique_ids), val_ratio, enroll_ratio)

    train_ids = set(unique_ids[:n_train])
    enroll_ids = set(unique_ids[n_train : n_train + n_enroll])
    val_ids = set(unique_ids[n_train + n_enroll : n_train + n_enroll + n_val])
    return train_ids, enroll_ids, val_ids


def split_train_val_ids(file_ids: np.ndarray, seed: int, val_ratio: float) -> tuple[set[str], set[str]]:
    unique_ids = unique_preserve_order(file_ids)
    if not unique_ids:
        return set(), set()
    if len(unique_ids) == 1:
        return set(unique_ids), set()

    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    n_val = max(1, int(round(len(unique_ids) * val_ratio)))
    n_val = min(n_val, len(unique_ids) - 1)
    val_ids = set(unique_ids[:n_val])
    train_ids = set(unique_ids[n_val:])
    return train_ids, val_ids


def build_joint_problem(
    train_raw: RawSamples,
    test_raw: RawSamples,
    enroll_action: str,
    val_ratio: float,
    enroll_ratio: float,
    seed: int,
) -> JointProblem:
    enroll_sub_labels = set(ACTION_TO_SUBLABELS[enroll_action])

    candidate_users = sorted(set(train_raw.users.tolist()) | set(test_raw.users.tolist()))
    eligible_users = [
        user
        for user in candidate_users
        if np.any((train_raw.users == user) & np.isin(train_raw.sub_labels, sorted(enroll_sub_labels)))
        and np.any((test_raw.users == user) & np.isin(test_raw.sub_labels, sorted(enroll_sub_labels)))
    ]
    if not eligible_users:
        raise ValueError(f"No eligible users found for action {enroll_action}.")

    user_to_index = {user: idx for idx, user in enumerate(eligible_users)}
    subaction_values = sorted({label for label in train_raw.sub_labels.tolist() if label != RANDOM_SUBLABEL})
    subaction_to_index = {label: idx for idx, label in enumerate(subaction_values)}

    train_identity_ids: set[str] = set()
    enroll_ids: set[str] = set()
    val_identity_ids: set[str] = set()

    for idx, user in enumerate(eligible_users):
        user_mask = (train_raw.users == user) & np.isin(train_raw.sub_labels, sorted(enroll_sub_labels))
        user_train_ids, user_enroll_ids, user_val_ids = split_positive_ids(
            file_ids=train_raw.file_ids[user_mask],
            seed=seed + idx,
            val_ratio=val_ratio,
            enroll_ratio=enroll_ratio,
        )
        train_identity_ids.update(user_train_ids)
        enroll_ids.update(user_enroll_ids)
        val_identity_ids.update(user_val_ids)

    other_action_train_ids: set[str] = set()
    other_action_val_ids: set[str] = set()
    other_action_mask = (train_raw.sub_labels != RANDOM_SUBLABEL) & ~np.isin(
        train_raw.sub_labels, sorted(enroll_sub_labels)
    )
    for idx, user in enumerate(sorted(set(train_raw.users.tolist()))):
        user_mask = other_action_mask & (train_raw.users == user)
        train_ids, val_ids = split_train_val_ids(
            file_ids=train_raw.file_ids[user_mask],
            seed=seed + 1000 + idx,
            val_ratio=val_ratio,
        )
        other_action_train_ids.update(train_ids)
        other_action_val_ids.update(val_ids)

    random_val_ids = set(train_raw.file_ids[train_raw.sub_labels == RANDOM_SUBLABEL].tolist())

    train_mask = np.isin(train_raw.file_ids, sorted(train_identity_ids | other_action_train_ids))
    enroll_mask = np.isin(train_raw.file_ids, sorted(enroll_ids))
    val_mask = np.isin(
        train_raw.file_ids,
        sorted(val_identity_ids | other_action_val_ids | random_val_ids),
    )
    test_mask = np.ones(len(test_raw.x), dtype=bool)

    train_raw_split = train_raw.subset(train_mask)
    identity_labels = np.full(len(train_raw_split.x), -1, dtype=np.int64)
    subaction_labels = np.array([subaction_to_index[label] for label in train_raw_split.sub_labels], dtype=np.int64)
    positive_train_mask = np.isin(train_raw_split.sub_labels, sorted(enroll_sub_labels)) & np.isin(
        train_raw_split.users, eligible_users
    )
    for user, idx in user_to_index.items():
        identity_labels[(train_raw_split.users == user) & positive_train_mask] = idx
    gate_labels = positive_train_mask.astype(np.int64)

    if not np.any(gate_labels == 1):
        raise ValueError("No enrolled-action positives found for training.")
    if not np.any(gate_labels == 0):
        raise ValueError("No non-enrolled-action negatives found for training.")

    train_split = JointTrainSplit(
        x=train_raw_split.x,
        identity_labels=identity_labels,
        gate_labels=gate_labels,
        subaction_labels=subaction_labels,
        users=train_raw_split.users,
        sub_labels=train_raw_split.sub_labels,
        file_ids=train_raw_split.file_ids,
    )
    enroll_raw = train_raw.subset(enroll_mask)
    val_raw = train_raw.subset(val_mask)
    test_raw_split = test_raw.subset(test_mask)

    return JointProblem(
        train_split=train_split,
        enroll_split=EvalSplit(
            x=enroll_raw.x,
            users=enroll_raw.users,
            sub_labels=enroll_raw.sub_labels,
            file_ids=enroll_raw.file_ids,
        ),
        val_split=EvalSplit(
            x=val_raw.x,
            users=val_raw.users,
            sub_labels=val_raw.sub_labels,
            file_ids=val_raw.file_ids,
        ),
        test_split=EvalSplit(
            x=test_raw_split.x,
            users=test_raw_split.users,
            sub_labels=test_raw_split.sub_labels,
            file_ids=test_raw_split.file_ids,
        ),
        eligible_users=eligible_users,
        user_to_index=user_to_index,
        subaction_to_index=subaction_to_index,
    )


def make_train_loader(
    split: JointTrainSplit,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    dataset = JointTrainDataset(split)
    pos_mask = split.gate_labels == 1
    neg_mask = split.gate_labels == 0

    weights = np.zeros(len(split.gate_labels), dtype=np.float32)
    if np.any(pos_mask):
        pos_scale = 0.5
        for label in np.unique(split.identity_labels[pos_mask]):
            class_mask = split.identity_labels == label
            weights[class_mask] = pos_scale / max(int(class_mask.sum()), 1) / max(
                len(np.unique(split.identity_labels[pos_mask])),
                1,
            )
    if np.any(neg_mask):
        weights[neg_mask] = 0.5 / max(int(neg_mask.sum()), 1)

    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(split.gate_labels),
        replacement=True,
        generator=generator,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)


def make_eval_loader(split: EvalSplit, batch_size: int, num_workers: int) -> DataLoader:
    dataset = EvalDataset(split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def scenario_name(sample_user: str, sample_sub_label: str, claimed_user: str, enroll_sub_labels: set[str]) -> str:
    if sample_user == claimed_user and sample_sub_label in enroll_sub_labels:
        return "target_enrolled"
    if sample_user == claimed_user and sample_sub_label == RANDOM_SUBLABEL:
        return "same_user_random"
    if sample_user == claimed_user:
        return "same_user_other_action"
    if sample_sub_label == RANDOM_SUBLABEL:
        return "other_user_random"
    if sample_sub_label in enroll_sub_labels:
        return "other_user_same_action"
    return "other_user_other_action"


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def compute_binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
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
    eer, eer_threshold = compute_eer(y_true, scores)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "far": float(far),
        "frr": float(frr),
        "tar": float(tar),
        "tnr": float(tnr),
        "balanced_acc": float(0.5 * (tar + tnr)),
        "hter": float(0.5 * (far + frr)),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "threshold": float(threshold),
        "pos_rate": float(y_true.mean()),
    }


def build_threshold_candidates(scores: np.ndarray, grid_size: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, grid_size)
    candidates = np.quantile(scores, quantiles)
    thresholds = np.unique(np.concatenate(([0.0, 1.0], candidates)))
    return np.sort(thresholds)[::-1]


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_far: float,
    grid_size: int = 128,
) -> ThresholdSelection:
    thresholds = build_threshold_candidates(scores, grid_size=grid_size)
    best_with_far: tuple[float, float, float] | None = None
    best_any: tuple[float, float, float] | None = None

    for threshold in thresholds:
        metrics = compute_binary_metrics(y_true, scores, float(threshold))
        candidate = (metrics["frr"], metrics["hter"], float(threshold))
        any_candidate = (
            abs(metrics["far"] - target_far) + metrics["hter"],
            metrics["hter"],
            float(threshold),
        )
        if metrics["far"] <= target_far + 1e-8:
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
    sample_users: np.ndarray,
    sample_sub_labels: np.ndarray,
    claimed_user: str,
    enroll_sub_labels: set[str],
    scores: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, float]]:
    decisions = (scores >= threshold).astype(np.int64)
    metrics: dict[str, dict[str, float]] = {}
    for scenario in SCENARIO_ORDER:
        mask = np.array(
            [
                scenario_name(user, sub_label, claimed_user, enroll_sub_labels) == scenario
                for user, sub_label in zip(sample_users, sample_sub_labels)
            ],
            dtype=bool,
        )
        if not np.any(mask):
            continue
        metrics[scenario] = {
            "count": float(mask.sum()),
            "accept_rate": float(decisions[mask].mean()),
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


def select_threshold_for_claim(
    split: EvalSplit,
    claimed_user: str,
    enroll_sub_labels: set[str],
    scores: np.ndarray,
    target_far: float,
    threshold_grid_size: int,
) -> ThresholdSelection:
    y_true = (
        (split.users == claimed_user) & np.isin(split.sub_labels, sorted(enroll_sub_labels))
    ).astype(np.int64)
    thresholds = build_threshold_candidates(scores, grid_size=threshold_grid_size)

    best_with_far: tuple[float, float, float, float] | None = None
    best_any: tuple[float, float, float, float] | None = None

    for threshold in thresholds:
        metrics = compute_binary_metrics(y_true, scores, float(threshold))
        scenarios = compute_scenario_metrics(
            sample_users=split.users,
            sample_sub_labels=split.sub_labels,
            claimed_user=claimed_user,
            enroll_sub_labels=enroll_sub_labels,
            scores=scores,
            threshold=float(threshold),
        )
        risk = scenario_risk(scenarios)
        candidate = (risk, metrics["frr"], metrics["hter"], float(threshold))
        any_candidate = (abs(metrics["far"] - target_far) + risk, risk, metrics["frr"], float(threshold))

        if metrics["far"] <= target_far + 1e-8:
            if best_with_far is None or candidate < best_with_far:
                best_with_far = candidate
        if best_any is None or any_candidate < best_any:
            best_any = any_candidate

    if best_with_far is not None:
        threshold = best_with_far[3]
        mode = "target_far_risk"
    elif best_any is not None:
        threshold = best_any[3]
        mode = "closest_far_risk"
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


def summarize_joint_split(name: str, split: JointTrainSplit) -> None:
    positives = int(np.sum(split.gate_labels == 1))
    negatives = int(np.sum(split.gate_labels == 0))
    print(
        f"[{name}] rows={len(split.gate_labels)} enrolled_pos={positives} other_action_neg={negatives}"
    )
    positive_users = len(np.unique(split.users[split.gate_labels == 1]))
    print(f"[{name}] positive_users={positive_users}")


def summarize_eval_split(name: str, split: EvalSplit, enroll_sub_labels: set[str]) -> None:
    enrolled = int(np.sum(np.isin(split.sub_labels, sorted(enroll_sub_labels))))
    other_action = int(
        np.sum((split.sub_labels != RANDOM_SUBLABEL) & ~np.isin(split.sub_labels, sorted(enroll_sub_labels)))
    )
    random_count = int(np.sum(split.sub_labels == RANDOM_SUBLABEL))
    print(
        f"[{name}] rows={len(split.x)} enrolled={enrolled} other_action={other_action} random={random_count}"
    )


def center_pull_loss(embeddings: torch.Tensor, identity_labels: torch.Tensor) -> torch.Tensor:
    positive_mask = identity_labels >= 0
    if positive_mask.sum() == 0:
        return torch.zeros((), device=embeddings.device)

    embeddings = F.normalize(embeddings[positive_mask], dim=1)
    labels = identity_labels[positive_mask]
    losses: list[torch.Tensor] = []
    for label in labels.unique():
        mask = labels == label
        if mask.sum() < 2:
            continue
        center = F.normalize(embeddings[mask].mean(dim=0, keepdim=True), dim=1)
        cosine = (embeddings[mask] * center).sum(dim=1)
        losses.append(1.0 - cosine.mean())
    if not losses:
        return torch.zeros((), device=embeddings.device)
    return torch.stack(losses).mean()


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    identity_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    positive_mask = identity_labels >= 0
    if positive_mask.sum() < 2:
        return torch.zeros((), device=embeddings.device)

    embeddings = F.normalize(embeddings[positive_mask], dim=1)
    labels = identity_labels[positive_mask]

    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_pairs = label_matrix & ~self_mask
    if not positive_pairs.any():
        return torch.zeros((), device=embeddings.device)

    logits = embeddings @ embeddings.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    mean_log_prob_pos = (positive_pairs.float() * log_prob).sum(dim=1) / (
        positive_pairs.float().sum(dim=1) + 1e-12
    )
    valid_rows = positive_pairs.sum(dim=1) > 0
    if not valid_rows.any():
        return torch.zeros((), device=embeddings.device)
    return -mean_log_prob_pos[valid_rows].mean()


def orthogonality_loss(common_proj: torch.Tensor, diff_feat: torch.Tensor) -> torch.Tensor:
    common_proj = F.normalize(common_proj, dim=1)
    diff_feat = F.normalize(diff_feat, dim=1)
    cosine = torch.sum(common_proj * diff_feat, dim=1)
    return torch.mean(cosine.pow(2))


def augment_time_series(
    x: torch.Tensor,
    noise_std: float,
    scale_std: float,
    max_shift: int,
) -> torch.Tensor:
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std

    if scale_std > 0:
        scale = 1.0 + torch.randn(x.size(0), x.size(1), 1, device=x.device) * scale_std
        x = x * torch.clamp(scale, min=0.7, max=1.3)

    if max_shift > 0:
        shifts = torch.randint(-max_shift, max_shift + 1, (x.size(0),), device=x.device)
        if torch.any(shifts != 0):
            shifted = []
            for sample, shift in zip(x, shifts):
                shifted.append(torch.roll(sample, shifts=int(shift.item()), dims=-1))
            x = torch.stack(shifted, dim=0)

    return x


def compute_joint_loss(
    output: dict[str, torch.Tensor],
    identity_labels: torch.Tensor,
    gate_labels: torch.Tensor,
    subaction_labels: torch.Tensor,
    common_consistency_weight: float,
    identity_label_smoothing: float,
    contrastive_temperature: float,
    grl_identity_weight: float,
    orthogonality_weight: float,
    subaction_weight: float,
    use_arcface: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses: dict[str, torch.Tensor] = {}
    losses["reconstruction"] = F.mse_loss(output["reconstructed"], output["backbone_feat"])
    losses["action_gate"] = F.cross_entropy(output["action_logits"], gate_labels)
    losses["subaction_ce"] = F.cross_entropy(
        output["subaction_logits"],
        subaction_labels,
        label_smoothing=identity_label_smoothing,
    )

    positive_mask = identity_labels >= 0
    if positive_mask.sum() > 0:
        identity_logits = output["identity_margin_logits"] if use_arcface else output["logits"]
        losses["identity_ce"] = F.cross_entropy(
            identity_logits[positive_mask],
            identity_labels[positive_mask],
            label_smoothing=identity_label_smoothing,
        )
        positive_labels = identity_labels[positive_mask]
        pairs = create_pair_indices(batch_size=int(positive_labels.numel()), labels=positive_labels)
        if pairs is not None and len(pairs) > 0:
            positive_common = output["common"][positive_mask]
            losses["common_consistency"] = F.mse_loss(
                positive_common[pairs[:, 0]],
                positive_common[pairs[:, 1]],
            )
        else:
            losses["common_consistency"] = torch.zeros((), device=identity_labels.device)
        losses["center_pull"] = center_pull_loss(output["diff"], identity_labels)
        losses["supcon"] = supervised_contrastive_loss(
            output["diff"],
            identity_labels,
            temperature=contrastive_temperature,
        )
        losses["common_adv_identity"] = F.cross_entropy(
            output["common_adv_logits"][positive_mask],
            identity_labels[positive_mask],
        )
        losses["orthogonality"] = orthogonality_loss(
            output["common_diff_proj"][positive_mask],
            output["diff"][positive_mask],
        )
    else:
        zero = torch.zeros((), device=identity_labels.device)
        losses["identity_ce"] = zero
        losses["common_consistency"] = zero
        losses["center_pull"] = zero
        losses["supcon"] = zero
        losses["common_adv_identity"] = zero
        losses["orthogonality"] = zero

    common_std = torch.std(output["common"], dim=0, unbiased=False).mean()
    losses["anti_collapse"] = -torch.clamp(common_std, min=1e-6)
    losses["residual_reg"] = torch.mean(torch.clamp(torch.abs(output["residual"]), max=10.0))

    total_loss = (
        1.0 * losses["reconstruction"]
        + 1.0 * losses["action_gate"]
        + subaction_weight * losses["subaction_ce"]
        + 1.0 * losses["identity_ce"]
        + common_consistency_weight * losses["common_consistency"]
        + 0.2 * losses["center_pull"]
        + 0.25 * losses["supcon"]
        + grl_identity_weight * losses["common_adv_identity"]
        + orthogonality_weight * losses["orthogonality"]
        + 0.1 * losses["anti_collapse"]
        + 0.01 * losses["residual_reg"]
    )
    return total_loss, losses


def train_epoch(
    model: JointDiffAuthVerifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    common_consistency_weight: float,
    identity_label_smoothing: float,
    contrastive_temperature: float,
    aug_noise_std: float,
    aug_scale_std: float,
    aug_max_shift: int,
    grl_lambda: float,
    grl_identity_weight: float,
    orthogonality_weight: float,
    subaction_weight: float,
    use_arcface: bool,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "reconstruction": 0.0,
        "action_gate": 0.0,
        "subaction_ce": 0.0,
        "identity_ce": 0.0,
        "center_pull": 0.0,
        "supcon": 0.0,
        "common_adv_identity": 0.0,
        "orthogonality": 0.0,
        "samples": 0.0,
    }

    for x, identity_labels, gate_labels, subaction_labels in loader:
        x = x.to(device)
        identity_labels = identity_labels.to(device)
        gate_labels = gate_labels.to(device)
        subaction_labels = subaction_labels.to(device)
        x = augment_time_series(
            x,
            noise_std=aug_noise_std,
            scale_std=aug_scale_std,
            max_shift=aug_max_shift,
        )

        margin_labels = identity_labels.clone()
        margin_labels[margin_labels < 0] = 0
        output = model(x, grl_lambda=grl_lambda, identity_labels=margin_labels)
        total_loss, losses = compute_joint_loss(
            output=output,
            identity_labels=identity_labels,
            gate_labels=gate_labels,
            subaction_labels=subaction_labels,
            common_consistency_weight=common_consistency_weight,
            identity_label_smoothing=identity_label_smoothing,
            contrastive_temperature=contrastive_temperature,
            grl_identity_weight=grl_identity_weight,
            orthogonality_weight=orthogonality_weight,
            subaction_weight=subaction_weight,
            use_arcface=use_arcface,
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_size = len(gate_labels)
        totals["loss"] += float(total_loss.item()) * batch_size
        totals["reconstruction"] += float(losses["reconstruction"].item()) * batch_size
        totals["action_gate"] += float(losses["action_gate"].item()) * batch_size
        totals["subaction_ce"] += float(losses["subaction_ce"].item()) * batch_size
        totals["identity_ce"] += float(losses["identity_ce"].item()) * batch_size
        totals["center_pull"] += float(losses["center_pull"].item()) * batch_size
        totals["supcon"] += float(losses["supcon"].item()) * batch_size
        totals["common_adv_identity"] += float(losses["common_adv_identity"].item()) * batch_size
        totals["orthogonality"] += float(losses["orthogonality"].item()) * batch_size
        totals["samples"] += float(batch_size)

    denom = max(totals["samples"], 1.0)
    return {key: value / denom for key, value in totals.items() if key != "samples"}


def predict_embeddings_and_gate(
    model: JointDiffAuthVerifier,
    split: EvalSplit,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_eval_loader(split, batch_size=batch_size, num_workers=num_workers)
    embeddings_all: list[np.ndarray] = []
    gate_probs_all: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            output = model(x)
            embeddings = F.normalize(output["diff"], dim=1)
            gate_probs = F.softmax(output["action_logits"], dim=1)[:, 1]
            embeddings_all.append(embeddings.cpu().numpy())
            gate_probs_all.append(gate_probs.cpu().numpy())

    return np.concatenate(embeddings_all, axis=0), np.concatenate(gate_probs_all, axis=0)


def build_user_templates(
    split: EvalSplit,
    embeddings: np.ndarray,
    eligible_users: list[str],
) -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for user in eligible_users:
        mask = split.users == user
        if not np.any(mask):
            raise ValueError(f"Missing enrollment template samples for user {user}.")
        template = embeddings[mask].mean(axis=0)
        templates[user] = template / (np.linalg.norm(template) + 1e-12)
    return templates


def build_score_matrix(
    embeddings: np.ndarray,
    gate_probs: np.ndarray,
    templates: dict[str, np.ndarray],
    eligible_users: list[str],
    action_weight: float,
) -> np.ndarray:
    template_matrix = np.stack([templates[user] for user in eligible_users], axis=1)
    cosine_scores = np.clip(embeddings @ template_matrix, -1.0, 1.0)
    template_scores = (cosine_scores + 1.0) * 0.5
    return action_weight * gate_probs[:, None] + (1.0 - action_weight) * template_scores


def build_template_scores(
    embeddings: np.ndarray,
    templates: dict[str, np.ndarray],
    eligible_users: list[str],
) -> np.ndarray:
    template_matrix = np.stack([templates[user] for user in eligible_users], axis=1)
    cosine_scores = np.clip(embeddings @ template_matrix, -1.0, 1.0)
    return (cosine_scores + 1.0) * 0.5


def fuse_score_matrix(
    template_scores: np.ndarray,
    gate_probs: np.ndarray,
    action_weight: float,
) -> np.ndarray:
    return action_weight * gate_probs[:, None] + (1.0 - action_weight) * template_scores


def calibrate_thresholds(
    split: EvalSplit,
    score_matrix: np.ndarray,
    eligible_users: list[str],
    enroll_sub_labels: set[str],
    target_far: float,
    threshold_grid_size: int,
) -> dict[str, ThresholdSelection]:
    thresholds: dict[str, ThresholdSelection] = {}
    for user_idx, claimed_user in enumerate(eligible_users):
        thresholds[claimed_user] = select_threshold_for_claim(
            split=split,
            claimed_user=claimed_user,
            enroll_sub_labels=enroll_sub_labels,
            scores=score_matrix[:, user_idx],
            target_far=target_far,
            threshold_grid_size=threshold_grid_size,
        )
    return thresholds


def evaluate_claims(
    split: EvalSplit,
    score_matrix: np.ndarray,
    eligible_users: list[str],
    enroll_sub_labels: set[str],
    thresholds: dict[str, ThresholdSelection],
    report_users: list[str],
) -> tuple[list[dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    results: list[dict[str, float]] = []
    scenario_by_user: dict[str, dict[str, dict[str, float]]] = {}
    for user_idx, claimed_user in enumerate(eligible_users):
        if report_users and claimed_user not in report_users:
            continue
        scores = score_matrix[:, user_idx]
        threshold = thresholds[claimed_user].threshold
        y_true = (
            (split.users == claimed_user) & np.isin(split.sub_labels, sorted(enroll_sub_labels))
        ).astype(np.int64)
        metrics = compute_binary_metrics(y_true, scores, threshold)
        scenario_metrics = compute_scenario_metrics(
            sample_users=split.users,
            sample_sub_labels=split.sub_labels,
            claimed_user=claimed_user,
            enroll_sub_labels=enroll_sub_labels,
            scores=scores,
            threshold=threshold,
        )
        scenario_by_user[claimed_user] = scenario_metrics
        results.append(
            {
                "user": claimed_user,
                **metrics,
                "risk": scenario_risk(scenario_metrics),
            }
        )
    return results, scenario_by_user


def select_action_weight(
    split: EvalSplit,
    template_scores: np.ndarray,
    gate_probs: np.ndarray,
    eligible_users: list[str],
    enroll_sub_labels: set[str],
    target_far: float,
    threshold_grid_size: int,
    candidate_weights: list[float],
    report_users: list[str],
) -> tuple[float, dict[str, ThresholdSelection], list[dict[str, float]], float, float, float]:
    best_tuple: tuple[float, float, float, float] | None = None
    best_weight = candidate_weights[0]
    best_thresholds: dict[str, ThresholdSelection] | None = None
    best_results: list[dict[str, float]] | None = None

    for action_weight in candidate_weights:
        score_matrix = fuse_score_matrix(template_scores, gate_probs, action_weight)
        thresholds = calibrate_thresholds(
            split=split,
            score_matrix=score_matrix,
            eligible_users=eligible_users,
            enroll_sub_labels=enroll_sub_labels,
            target_far=target_far,
            threshold_grid_size=threshold_grid_size,
        )
        results, _ = evaluate_claims(
            split=split,
            score_matrix=score_matrix,
            eligible_users=eligible_users,
            enroll_sub_labels=enroll_sub_labels,
            thresholds=thresholds,
            report_users=report_users,
        )
        mean_risk = float(np.mean([result["risk"] for result in results]))
        mean_far = float(np.mean([result["far"] for result in results]))
        mean_frr = float(np.mean([result["frr"] for result in results]))
        candidate = (mean_risk, mean_frr, mean_far, action_weight)

        if best_tuple is None or candidate < best_tuple:
            best_tuple = candidate
            best_weight = action_weight
            best_thresholds = copy.deepcopy(thresholds)
            best_results = copy.deepcopy(results)

    if best_tuple is None or best_thresholds is None or best_results is None:
        raise RuntimeError("Failed to select action weight.")

    return (
        best_weight,
        best_thresholds,
        best_results,
        best_tuple[0],
        best_tuple[2],
        best_tuple[1],
    )


def format_scenarios(scenario_metrics: dict[str, dict[str, float]]) -> str:
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
    enroll_action: str,
    model_state: dict[str, torch.Tensor],
    templates: dict[str, np.ndarray],
    thresholds: dict[str, ThresholdSelection],
    problem: JointProblem,
    args: argparse.Namespace,
    action_weight: float,
) -> None:
    if not out_dir:
        return
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"joint_verifier_{enroll_action}.pt"
    artifact = {
        "enroll_action": enroll_action,
        "enroll_sub_labels": list(ACTION_TO_SUBLABELS[enroll_action]),
        "eligible_users": problem.eligible_users,
        "user_to_index": problem.user_to_index,
        "thresholds": {user: thresholds[user].threshold for user in problem.eligible_users},
        "templates": {user: torch.from_numpy(template.astype(np.float32)) for user, template in templates.items()},
        "action_weight": float(action_weight),
        "model_state_dict": model_state,
        "model_config": {
            "feat_dim": args.feat_dim,
            "common_dim": args.common_dim,
            "diff_dim": args.diff_dim,
            "num_users": len(problem.eligible_users),
            "num_subactions": len(problem.subaction_to_index),
        },
    }
    torch.save(artifact, artifact_path)

    metadata_path = artifact_path.with_suffix(".json")
    metadata = {
        "enroll_action": enroll_action,
        "eligible_users": problem.eligible_users,
        "thresholds": {user: float(thresholds[user].threshold) for user in problem.eligible_users},
        "action_weight": float(action_weight),
        "target_far": float(args.target_far),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[Artifact] saved={artifact_path}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_epochs = max(0, warmup_epochs)

    def lr_lambda(epoch_idx: int) -> float:
        epoch = epoch_idx + 1
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            return max(epoch / warmup_epochs, 1e-3)

        if total_epochs <= warmup_epochs:
            return 1.0

        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint multi-user DiffAuth training for action-aware biometric verification."
    )
    parser.add_argument(
        "--enroll-action",
        required=True,
        choices=sorted(ACTION_TO_SUBLABELS.keys()),
        help="Registered unlock gesture to accept.",
    )
    parser.add_argument("--target-user", help="Optional user id to print focused report for.")
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--common-dim", type=int, default=64)
    parser.add_argument("--diff-dim", type=int, default=64)
    parser.add_argument("--action-weight", type=float, default=0.40)
    parser.add_argument(
        "--action-weight-candidates",
        default="0.25,0.35,0.45,0.55",
        help="Comma-separated fusion weights searched on validation. Use a single value to disable search.",
    )
    parser.add_argument("--common-consistency-weight", type=float, default=0.5)
    parser.add_argument("--identity-label-smoothing", type=float, default=0.05)
    parser.add_argument("--contrastive-temperature", type=float, default=0.2)
    parser.add_argument("--grl-lambda", type=float, default=0.3)
    parser.add_argument("--grl-identity-weight", type=float, default=0.1)
    parser.add_argument("--orthogonality-weight", type=float, default=0.1)
    parser.add_argument("--subaction-weight", type=float, default=0.0)
    parser.add_argument("--use-arcface", action="store_true", help="Use ArcFace-style identity head during training.")
    parser.add_argument("--aug-noise-std", type=float, default=0.015)
    parser.add_argument("--aug-scale-std", type=float, default=0.05)
    parser.add_argument("--aug-max-shift", type=int, default=4)
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--threshold-grid-size", type=int, default=96)
    parser.add_argument("--skip-validation", action="store_true", help="Disable validation during training and keep the last checkpoint.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--enroll-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact-dir", default="artifacts", help="Directory for saved artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    if not (0.0 < args.val_ratio < 0.5):
        raise ValueError("--val-ratio must be between 0 and 0.5.")
    if not (0.0 < args.enroll_ratio < 0.5):
        raise ValueError("--enroll-ratio must be between 0 and 0.5.")
    if not (0.0 < args.target_far < 1.0):
        raise ValueError("--target-far must be in (0, 1).")
    if not (0.0 <= args.action_weight <= 1.0):
        raise ValueError("--action-weight must be in [0, 1].")
    if not (0.0 <= args.identity_label_smoothing < 1.0):
        raise ValueError("--identity-label-smoothing must be in [0, 1).")
    if args.contrastive_temperature <= 0.0:
        raise ValueError("--contrastive-temperature must be > 0.")
    if (
        args.grl_lambda < 0.0
        or args.grl_identity_weight < 0.0
        or args.orthogonality_weight < 0.0
        or args.subaction_weight < 0.0
    ):
        raise ValueError("GRL/orthogonality/subaction parameters must be non-negative.")
    if args.aug_noise_std < 0.0 or args.aug_scale_std < 0.0 or args.aug_max_shift < 0:
        raise ValueError("augmentation parameters must be non-negative.")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be >= 1.")
    if args.threshold_grid_size < 8:
        raise ValueError("--threshold-grid-size must be >= 8.")
    if args.weight_decay < 0.0 or args.warmup_epochs < 0:
        raise ValueError("--weight-decay and --warmup-epochs must be non-negative.")
    if not (0.0 < args.min_lr_ratio <= 1.0):
        raise ValueError("--min-lr-ratio must be in (0, 1].")
    action_weight_candidates = [float(item.strip()) for item in args.action_weight_candidates.split(",") if item.strip()]
    if not action_weight_candidates:
        action_weight_candidates = [args.action_weight]
    for weight in action_weight_candidates:
        if not (0.0 <= weight <= 1.0):
            raise ValueError("--action-weight-candidates values must be in [0, 1].")

    train_raw = load_raw_from_files(args.train_files)
    test_raw = load_raw_from_files(args.test_files)
    problem = build_joint_problem(
        train_raw=train_raw,
        test_raw=test_raw,
        enroll_action=args.enroll_action,
        val_ratio=args.val_ratio,
        enroll_ratio=args.enroll_ratio,
        seed=args.seed,
    )
    enroll_sub_labels = set(ACTION_TO_SUBLABELS[args.enroll_action])
    report_users = [str(args.target_user)] if args.target_user else problem.eligible_users
    if args.target_user and str(args.target_user) not in problem.eligible_users:
        raise ValueError(f"Target user {args.target_user} is not eligible for {args.enroll_action}.")

    print(
        f"[Setup] enroll_action={args.enroll_action} eligible_users={len(problem.eligible_users)} "
        f"train_rows={len(problem.train_split.x)} val_rows={len(problem.val_split.x)} "
        f"test_rows={len(problem.test_split.x)}"
    )
    print(f"[Setup] action_weight_candidates={action_weight_candidates}")
    summarize_joint_split("Train", problem.train_split)
    summarize_eval_split("Enroll", problem.enroll_split, enroll_sub_labels)
    summarize_eval_split("Val", problem.val_split, enroll_sub_labels)
    summarize_eval_split("Test", problem.test_split, enroll_sub_labels)

    train_loader = make_train_loader(
        split=problem.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model = JointDiffAuthVerifier(
        feat_dim=args.feat_dim,
        common_dim=args.common_dim,
        diff_dim=args.diff_dim,
        num_users=len(problem.eligible_users),
        num_subactions=len(problem.subaction_to_index),
    ).to(device)
    model.use_arcface = args.use_arcface
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(
        optimizer=optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_risk = float("inf")
    best_templates: dict[str, np.ndarray] | None = None
    best_thresholds: dict[str, ThresholdSelection] | None = None
    best_val_results: list[dict[str, float]] | None = None
    last_val_risk: float | None = None
    best_action_weight = args.action_weight

    for epoch in range(1, args.epochs + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_stats = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            common_consistency_weight=args.common_consistency_weight,
            identity_label_smoothing=args.identity_label_smoothing,
            contrastive_temperature=args.contrastive_temperature,
            aug_noise_std=args.aug_noise_std,
            aug_scale_std=args.aug_scale_std,
            aug_max_shift=args.aug_max_shift,
            grl_lambda=args.grl_lambda,
            grl_identity_weight=args.grl_identity_weight,
            orthogonality_weight=args.orthogonality_weight,
            subaction_weight=args.subaction_weight,
            use_arcface=args.use_arcface,
        )

        if args.skip_validation:
            best_state = copy.deepcopy(model.state_dict())
            print(
                f"[Epoch {epoch:03d}] loss={train_stats['loss']:.4f} "
                f"lr={current_lr:.6f} "
                f"recon={train_stats['reconstruction']:.4f} gate={train_stats['action_gate']:.4f} "
                f"sub={train_stats['subaction_ce']:.4f} id={train_stats['identity_ce']:.4f} "
                f"center={train_stats['center_pull']:.4f} "
                f"supcon={train_stats['supcon']:.4f} adv={train_stats['common_adv_identity']:.4f} "
                f"ortho={train_stats['orthogonality']:.4f} val=disabled"
            )
            scheduler.step()
            continue

        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if not should_eval:
            print(
                f"[Epoch {epoch:03d}] loss={train_stats['loss']:.4f} "
                f"lr={current_lr:.6f} "
                f"recon={train_stats['reconstruction']:.4f} gate={train_stats['action_gate']:.4f} "
                f"sub={train_stats['subaction_ce']:.4f} id={train_stats['identity_ce']:.4f} "
                f"center={train_stats['center_pull']:.4f} "
                f"supcon={train_stats['supcon']:.4f} adv={train_stats['common_adv_identity']:.4f} "
                f"ortho={train_stats['orthogonality']:.4f} val=skipped"
            )
            scheduler.step()
            continue

        enroll_embeddings, _ = predict_embeddings_and_gate(
            model=model,
            split=problem.enroll_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        templates = build_user_templates(
            split=problem.enroll_split,
            embeddings=enroll_embeddings,
            eligible_users=problem.eligible_users,
        )

        val_embeddings, val_gate_probs = predict_embeddings_and_gate(
            model=model,
            split=problem.val_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        val_template_scores = build_template_scores(
            embeddings=val_embeddings,
            templates=templates,
            eligible_users=problem.eligible_users,
        )
        (
            selected_action_weight,
            val_thresholds,
            val_results,
            mean_val_risk,
            mean_val_far,
            mean_val_frr,
        ) = select_action_weight(
            split=problem.val_split,
            template_scores=val_template_scores,
            gate_probs=val_gate_probs,
            eligible_users=problem.eligible_users,
            enroll_sub_labels=enroll_sub_labels,
            target_far=args.target_far,
            threshold_grid_size=args.threshold_grid_size,
            candidate_weights=action_weight_candidates,
            report_users=problem.eligible_users,
        )
        last_val_risk = mean_val_risk

        if mean_val_risk < best_val_risk:
            best_val_risk = mean_val_risk
            best_state = copy.deepcopy(model.state_dict())
            best_templates = {user: np.array(template, copy=True) for user, template in templates.items()}
            best_thresholds = copy.deepcopy(val_thresholds)
            best_val_results = copy.deepcopy(val_results)
            best_action_weight = selected_action_weight

        print(
            f"[Epoch {epoch:03d}] loss={train_stats['loss']:.4f} "
            f"lr={current_lr:.6f} "
            f"recon={train_stats['reconstruction']:.4f} gate={train_stats['action_gate']:.4f} "
            f"sub={train_stats['subaction_ce']:.4f} id={train_stats['identity_ce']:.4f} "
            f"center={train_stats['center_pull']:.4f} "
            f"supcon={train_stats['supcon']:.4f} adv={train_stats['common_adv_identity']:.4f} "
            f"ortho={train_stats['orthogonality']:.4f} w={selected_action_weight:.2f} "
            f"val_far={mean_val_far:.4f} val_frr={mean_val_frr:.4f} val_risk={mean_val_risk:.4f}"
        )
        scheduler.step()

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    if args.skip_validation:
        print("[Setup] validation=disabled thresholding=fixed_0.5")
        model.load_state_dict(best_state)
        enroll_embeddings, _ = predict_embeddings_and_gate(
            model=model,
            split=problem.enroll_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        best_templates = build_user_templates(
            split=problem.enroll_split,
            embeddings=enroll_embeddings,
            eligible_users=problem.eligible_users,
        )
        best_thresholds = {
            user: ThresholdSelection(
                threshold=0.5,
                eer=float("nan"),
                eer_threshold=float("nan"),
                far=float("nan"),
                frr=float("nan"),
                hter=float("nan"),
                mode="fixed_0.5",
            )
            for user in problem.eligible_users
        }
        best_val_risk = float("nan")
        last_val_risk = float("nan")
        best_action_weight = args.action_weight
    elif best_templates is None or best_thresholds is None or best_val_results is None:
        raise RuntimeError("Training finished without a valid validation checkpoint.")

    model.load_state_dict(best_state)
    test_embeddings, test_gate_probs = predict_embeddings_and_gate(
        model=model,
        split=problem.test_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    test_template_scores = build_template_scores(
        embeddings=test_embeddings,
        templates=best_templates,
        eligible_users=problem.eligible_users,
    )
    test_score_matrix = fuse_score_matrix(
        template_scores=test_template_scores,
        gate_probs=test_gate_probs,
        action_weight=best_action_weight,
    )
    test_results, test_scenarios = evaluate_claims(
        split=problem.test_split,
        score_matrix=test_score_matrix,
        eligible_users=problem.eligible_users,
        enroll_sub_labels=enroll_sub_labels,
        thresholds=best_thresholds,
        report_users=report_users,
    )

    maybe_save_artifact(
        out_dir=args.artifact_dir,
        enroll_action=args.enroll_action,
        model_state=best_state,
        templates=best_templates,
        thresholds=best_thresholds,
        problem=problem,
        args=args,
        action_weight=best_action_weight,
    )

    print("\n[Summary By User]")
    for result in test_results:
        user = result["user"]
        print(
            f"user={user} acc={result['acc']:.4f} f1={result['f1']:.4f} "
            f"far={result['far']:.4f} frr={result['frr']:.4f} "
            f"eer={result['eer']:.4f} thr={result['threshold']:.4f} risk={result['risk']:.4f}"
        )
        print(f"[Scenarios][{user}] {format_scenarios(test_scenarios[user])}")

    mean_acc = float(np.mean([result["acc"] for result in test_results]))
    mean_f1 = float(np.mean([result["f1"] for result in test_results]))
    mean_far = float(np.mean([result["far"] for result in test_results]))
    mean_frr = float(np.mean([result["frr"] for result in test_results]))
    mean_eer = float(np.mean([result["eer"] for result in test_results]))
    mean_risk = float(np.mean([result["risk"] for result in test_results]))
    print("\n[Summary Mean]")
    print(
        f"users={len(test_results)} mean_acc={mean_acc:.4f} mean_f1={mean_f1:.4f} "
        f"mean_far={mean_far:.4f} mean_frr={mean_frr:.4f} mean_eer={mean_eer:.4f} "
        f"mean_risk={mean_risk:.4f} best_val_risk={best_val_risk:.4f} "
        f"last_val_risk={last_val_risk if last_val_risk is not None else float('nan'):.4f} "
        f"best_action_weight={best_action_weight:.2f}"
    )


if __name__ == "__main__":
    main()

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

    def subset(self, mask: np.ndarray) -> "RawSamples":
        return RawSamples(
            x=self.x[mask],
            users=self.users[mask],
            sub_labels=self.sub_labels[mask],
            file_ids=self.file_ids[mask],
        )


@dataclass
class JointTrainSplit:
    x: np.ndarray
    identity_labels: np.ndarray
    gate_labels: np.ndarray
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


class JointTrainDataset(Dataset):
    def __init__(self, split: JointTrainSplit):
        self.x = torch.from_numpy(split.x).float()
        self.identity_labels = torch.from_numpy(split.identity_labels).long()
        self.gate_labels = torch.from_numpy(split.gate_labels).long()

    def __len__(self) -> int:
        return len(self.gate_labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.identity_labels[idx], self.gate_labels[idx]


class EvalDataset(Dataset):
    def __init__(self, split: EvalSplit):
        self.x = torch.from_numpy(split.x).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


class ImuTensorBackbone(nn.Module):
    """From github.com/matisiekpl/imutensor"""

    def __init__(self, input_size: int = 2, hidden_size: int = 22):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [B, 128, 2]
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]


class HARCNNBackbone(nn.Module):
    """From github.com/jchiang2/Human-Activity-Recognition"""

    def __init__(self, input_size: int = 2, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_size, 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
        )
        self.embedding = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 116, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.embedding(x)


class DeepConvLSTMBackbone(nn.Module):
    """From github.com/dspanah/Sensor-Based-Human-Activity-Recognition-DeepConvLSTM-Pytorch"""

    def __init__(self, input_channels: int = 2, n_filters: int = 64, n_hidden: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, n_filters, 5)
        self.conv2 = nn.Conv1d(n_filters, n_filters, 5)
        self.conv3 = nn.Conv1d(n_filters, n_filters, 5)
        self.conv4 = nn.Conv1d(n_filters, n_filters, 5)
        self.lstm1 = nn.LSTM(n_filters, n_hidden, batch_first=False)
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, batch_first=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.permute(2, 0, 1).contiguous()
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x[-1]


class JointVerifier(nn.Module):
    def __init__(self, backbone: nn.Module, embedding_dim: int, num_users: int):
        super().__init__()
        self.backbone = backbone
        self.action_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 2),
        )
        self.identity_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_users),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.backbone(x)
        embedding = F.normalize(embedding, dim=1)
        return {
            "embedding": embedding,
            "action_logits": self.action_head(embedding),
            "identity_logits": self.identity_head(embedding),
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

    for fp in file_paths:
        x, meta = load_time_series_with_meta(fp)
        prefix = Path(fp).stem
        file_ids = np.array([f"{prefix}:{fid}" for fid in meta["file_ids"]], dtype=str)
        xs.append(x)
        users_list.append(meta["users"].astype(str))
        sub_labels_list.append(meta["sub_labels"].astype(str))
        file_ids_list.append(file_ids)

    return RawSamples(
        x=np.concatenate(xs, axis=0),
        users=np.concatenate(users_list, axis=0),
        sub_labels=np.concatenate(sub_labels_list, axis=0),
        file_ids=np.concatenate(file_ids_list, axis=0),
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
    return set(unique_ids[n_val:]), set(unique_ids[:n_val])


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
        raise ValueError(f"No eligible users for {enroll_action}.")
    user_to_index = {user: idx for idx, user in enumerate(eligible_users)}

    train_identity_ids: set[str] = set()
    enroll_ids: set[str] = set()
    val_identity_ids: set[str] = set()
    for idx, user in enumerate(eligible_users):
        user_mask = (train_raw.users == user) & np.isin(train_raw.sub_labels, sorted(enroll_sub_labels))
        train_ids, user_enroll_ids, user_val_ids = split_positive_ids(
            train_raw.file_ids[user_mask],
            seed=seed + idx,
            val_ratio=val_ratio,
            enroll_ratio=enroll_ratio,
        )
        train_identity_ids.update(train_ids)
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
            train_raw.file_ids[user_mask], seed=seed + 1000 + idx, val_ratio=val_ratio
        )
        other_action_train_ids.update(train_ids)
        other_action_val_ids.update(val_ids)

    random_val_ids = set(train_raw.file_ids[train_raw.sub_labels == RANDOM_SUBLABEL].tolist())

    train_mask = np.isin(train_raw.file_ids, sorted(train_identity_ids | other_action_train_ids))
    enroll_mask = np.isin(train_raw.file_ids, sorted(enroll_ids))
    val_mask = np.isin(train_raw.file_ids, sorted(val_identity_ids | other_action_val_ids | random_val_ids))
    test_mask = np.ones(len(test_raw.x), dtype=bool)

    train_raw_split = train_raw.subset(train_mask)
    identity_labels = np.full(len(train_raw_split.x), -1, dtype=np.int64)
    positive_mask = np.isin(train_raw_split.sub_labels, sorted(enroll_sub_labels)) & np.isin(
        train_raw_split.users, eligible_users
    )
    for user, idx in user_to_index.items():
        identity_labels[(train_raw_split.users == user) & positive_mask] = idx
    gate_labels = positive_mask.astype(np.int64)

    return JointProblem(
        train_split=JointTrainSplit(
            x=train_raw_split.x,
            identity_labels=identity_labels,
            gate_labels=gate_labels,
            users=train_raw_split.users,
            sub_labels=train_raw_split.sub_labels,
            file_ids=train_raw_split.file_ids,
        ),
        enroll_split=EvalSplit(
            x=train_raw.subset(enroll_mask).x,
            users=train_raw.subset(enroll_mask).users,
            sub_labels=train_raw.subset(enroll_mask).sub_labels,
            file_ids=train_raw.subset(enroll_mask).file_ids,
        ),
        val_split=EvalSplit(
            x=train_raw.subset(val_mask).x,
            users=train_raw.subset(val_mask).users,
            sub_labels=train_raw.subset(val_mask).sub_labels,
            file_ids=train_raw.subset(val_mask).file_ids,
        ),
        test_split=EvalSplit(
            x=test_raw.subset(test_mask).x,
            users=test_raw.subset(test_mask).users,
            sub_labels=test_raw.subset(test_mask).sub_labels,
            file_ids=test_raw.subset(test_mask).file_ids,
        ),
        eligible_users=eligible_users,
        user_to_index=user_to_index,
    )


def make_train_loader(split: JointTrainSplit, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    dataset = JointTrainDataset(split)
    pos_mask = split.gate_labels == 1
    neg_mask = split.gate_labels == 0
    weights = np.zeros(len(split.gate_labels), dtype=np.float32)
    positive_users = np.unique(split.identity_labels[pos_mask])
    if np.any(pos_mask):
        for label in positive_users:
            class_mask = split.identity_labels == label
            weights[class_mask] = 0.5 / max(int(class_mask.sum()), 1) / max(len(positive_users), 1)
    if np.any(neg_mask):
        weights[neg_mask] = 0.5 / max(int(neg_mask.sum()), 1)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), len(split.gate_labels), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)


def make_eval_loader(split: EvalSplit, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(EvalDataset(split), batch_size=batch_size, shuffle=False, num_workers=num_workers)


def center_pull_loss(embeddings: torch.Tensor, identity_labels: torch.Tensor) -> torch.Tensor:
    positive_mask = identity_labels >= 0
    if positive_mask.sum() == 0:
        return torch.zeros((), device=embeddings.device)
    embeddings = embeddings[positive_mask]
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


def train_epoch(model: JointVerifier, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "gate": 0.0, "identity": 0.0, "center": 0.0, "samples": 0.0}
    for x, identity_labels, gate_labels in loader:
        x = x.to(device)
        identity_labels = identity_labels.to(device)
        gate_labels = gate_labels.to(device)
        output = model(x)
        gate_loss = F.cross_entropy(output["action_logits"], gate_labels)
        pos_mask = identity_labels >= 0
        if pos_mask.sum() > 0:
            identity_loss = F.cross_entropy(output["identity_logits"][pos_mask], identity_labels[pos_mask])
        else:
            identity_loss = torch.zeros((), device=device)
        center_loss = center_pull_loss(output["embedding"], identity_labels)
        loss = gate_loss + identity_loss + 0.2 * center_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = len(gate_labels)
        totals["loss"] += float(loss.item()) * batch_size
        totals["gate"] += float(gate_loss.item()) * batch_size
        totals["identity"] += float(identity_loss.item()) * batch_size
        totals["center"] += float(center_loss.item()) * batch_size
        totals["samples"] += float(batch_size)
    denom = max(totals["samples"], 1.0)
    return {k: v / denom for k, v in totals.items() if k != "samples"}


def predict_embeddings_and_gate(
    model: JointVerifier,
    split: EvalSplit,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_eval_loader(split, batch_size, num_workers)
    embeddings_all: list[np.ndarray] = []
    gate_probs_all: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            output = model(x)
            embeddings_all.append(output["embedding"].cpu().numpy())
            gate_probs_all.append(F.softmax(output["action_logits"], dim=1)[:, 1].cpu().numpy())
    return np.concatenate(embeddings_all, axis=0), np.concatenate(gate_probs_all, axis=0)


def build_user_templates(split: EvalSplit, embeddings: np.ndarray, eligible_users: list[str]) -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for user in eligible_users:
        mask = split.users == user
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
    }


def select_threshold(y_true: np.ndarray, scores: np.ndarray, target_far: float) -> ThresholdSelection:
    thresholds = np.unique(np.concatenate(([0.0, 1.0], scores)))
    thresholds = np.sort(thresholds)[::-1]
    best_with_far = None
    best_any = None
    for thr in thresholds:
        metrics = compute_binary_metrics(y_true, scores, float(thr))
        candidate = (metrics["frr"], metrics["hter"], float(thr))
        any_candidate = (abs(metrics["far"] - target_far) + metrics["hter"], metrics["hter"], float(thr))
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
    weighted_errors = []
    weights = []
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


def evaluate_claims(
    split: EvalSplit,
    score_matrix: np.ndarray,
    eligible_users: list[str],
    enroll_sub_labels: set[str],
    thresholds: dict[str, ThresholdSelection],
) -> tuple[list[dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    results = []
    scenarios = {}
    for user_idx, user in enumerate(eligible_users):
        scores = score_matrix[:, user_idx]
        y_true = ((split.users == user) & np.isin(split.sub_labels, sorted(enroll_sub_labels))).astype(np.int64)
        metrics = compute_binary_metrics(y_true, scores, thresholds[user].threshold)
        scenario_metrics = compute_scenario_metrics(
            split.users, split.sub_labels, user, enroll_sub_labels, scores, thresholds[user].threshold
        )
        results.append({"user": user, **metrics, "risk": scenario_risk(scenario_metrics)})
        scenarios[user] = scenario_metrics
    return results, scenarios


def format_scenarios(scenario_metrics: dict[str, dict[str, float]]) -> str:
    parts = []
    for scenario in SCENARIO_ORDER:
        if scenario not in scenario_metrics:
            continue
        info = scenario_metrics[scenario]
        parts.append(
            f"{scenario}:count={int(info['count'])},accept={info['accept_rate']:.4f},score={info['mean_score']:.4f}"
        )
    return " | ".join(parts)


def build_model(model_name: str, num_users: int) -> tuple[JointVerifier, int]:
    if model_name == "imutensor":
        embedding_dim = 22
        backbone = ImuTensorBackbone(input_size=2, hidden_size=embedding_dim)
    elif model_name == "har_cnn":
        embedding_dim = 128
        backbone = HARCNNBackbone(input_size=2, embedding_dim=embedding_dim)
    elif model_name == "deepconvlstm":
        embedding_dim = 128
        backbone = DeepConvLSTMBackbone(input_channels=2, n_hidden=embedding_dim)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return JointVerifier(backbone=backbone, embedding_dim=embedding_dim, num_users=num_users), embedding_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark selected GitHub IMU models on our unlock dataset.")
    parser.add_argument("--model", required=True, choices=["imutensor", "har_cnn", "deepconvlstm"])
    parser.add_argument("--enroll-action", required=True, choices=sorted(ACTION_TO_SUBLABELS.keys()))
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--action-weight", type=float, default=0.45)
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--enroll-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact-dir", default="artifacts_github_bench")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

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
    model, _ = build_model(args.model, num_users=len(problem.eligible_users))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = make_train_loader(problem.train_split, args.batch_size, args.num_workers, args.seed)

    print(
        f"[Setup] model={args.model} enroll_action={args.enroll_action} eligible_users={len(problem.eligible_users)} "
        f"train_rows={len(problem.train_split.x)} val_rows={len(problem.val_split.x)} test_rows={len(problem.test_split.x)}"
    )

    best_state = None
    best_templates = None
    best_thresholds = None
    best_val_risk = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(model, train_loader, optimizer, device)
        enroll_embeddings, _ = predict_embeddings_and_gate(model, problem.enroll_split, args.batch_size, args.num_workers, device)
        templates = build_user_templates(problem.enroll_split, enroll_embeddings, problem.eligible_users)
        val_embeddings, val_gate_probs = predict_embeddings_and_gate(model, problem.val_split, args.batch_size, args.num_workers, device)
        val_score_matrix = build_score_matrix(
            val_embeddings, val_gate_probs, templates, problem.eligible_users, args.action_weight
        )
        thresholds = {}
        for user_idx, user in enumerate(problem.eligible_users):
            y_true = ((problem.val_split.users == user) & np.isin(problem.val_split.sub_labels, sorted(enroll_sub_labels))).astype(np.int64)
            thresholds[user] = select_threshold(y_true, val_score_matrix[:, user_idx], args.target_far)
        val_results, _ = evaluate_claims(problem.val_split, val_score_matrix, problem.eligible_users, enroll_sub_labels, thresholds)
        mean_val_risk = float(np.mean([r["risk"] for r in val_results]))
        mean_val_far = float(np.mean([r["far"] for r in val_results]))
        mean_val_frr = float(np.mean([r["frr"] for r in val_results]))
        if mean_val_risk < best_val_risk:
            best_val_risk = mean_val_risk
            best_state = copy.deepcopy(model.state_dict())
            best_templates = {user: np.array(template, copy=True) for user, template in templates.items()}
            best_thresholds = copy.deepcopy(thresholds)
        print(
            f"[Epoch {epoch:03d}] loss={train_stats['loss']:.4f} gate={train_stats['gate']:.4f} "
            f"id={train_stats['identity']:.4f} center={train_stats['center']:.4f} "
            f"val_far={mean_val_far:.4f} val_frr={mean_val_frr:.4f} val_risk={mean_val_risk:.4f}"
        )

    if best_state is None or best_templates is None or best_thresholds is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    test_embeddings, test_gate_probs = predict_embeddings_and_gate(model, problem.test_split, args.batch_size, args.num_workers, device)
    test_score_matrix = build_score_matrix(
        test_embeddings, test_gate_probs, best_templates, problem.eligible_users, args.action_weight
    )
    test_results, test_scenarios = evaluate_claims(
        problem.test_split, test_score_matrix, problem.eligible_users, enroll_sub_labels, best_thresholds
    )

    output_dir = Path(args.artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.model}_{args.enroll_action}_summary.json"
    summary = {
        "model": args.model,
        "enroll_action": args.enroll_action,
        "best_val_risk": best_val_risk,
        "mean_metrics": {
            "acc": float(np.mean([r["acc"] for r in test_results])),
            "f1": float(np.mean([r["f1"] for r in test_results])),
            "far": float(np.mean([r["far"] for r in test_results])),
            "frr": float(np.mean([r["frr"] for r in test_results])),
            "eer": float(np.mean([r["eer"] for r in test_results])),
            "risk": float(np.mean([r["risk"] for r in test_results])),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[Summary Mean]")
    print(
        f"users={len(test_results)} mean_acc={summary['mean_metrics']['acc']:.4f} "
        f"mean_f1={summary['mean_metrics']['f1']:.4f} mean_far={summary['mean_metrics']['far']:.4f} "
        f"mean_frr={summary['mean_metrics']['frr']:.4f} mean_eer={summary['mean_metrics']['eer']:.4f} "
        f"mean_risk={summary['mean_metrics']['risk']:.4f} best_val_risk={best_val_risk:.4f}"
    )
    top_users = sorted(test_results, key=lambda item: item["risk"])[:3]
    print("\n[Sample Users]")
    for result in top_users:
        print(
            f"user={result['user']} acc={result['acc']:.4f} far={result['far']:.4f} "
            f"frr={result['frr']:.4f} risk={result['risk']:.4f}"
        )
        print(f"[Scenarios][{result['user']}] {format_scenarios(test_scenarios[result['user']])}")


if __name__ == "__main__":
    main()

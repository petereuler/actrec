"""
DiffAuth: 基于学习式残差的双路身份验证模块

核心思想：
- 共模特征路：学习不同人的共性特征
- 差模特征路：学习个人的独特特征
- 通过学习式残差实现两路的交互
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """A small stabilized MLP block used across common/diff branches."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualLearner(nn.Module):
    """带残差连接的残差学习器。"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 拼接后的特征 [B, feat_dim * 2]
               x[:, :feat_dim] 是原始特征 F
               x[:, feat_dim:] 是对齐后的共模特征 C_aligned

        Returns:
            r: 调制因子 [B, feat_dim], 范围 [-1, 1]
        """
        h = F.relu(self.fc1(x))
        # 第二层 + 残差连接（取原始特征 F 作为残差）
        r = torch.tanh(self.fc2(h) + x[:, : self.output_dim])
        return r


class LearnableResidualModule(nn.Module):
    """
    学习式残差模块。

    分解过程：
    F -> CommonEncoder -> C -> CommonAligner -> C_aligned
    [F; C_aligned] -> ResidualLearner -> r
    D_modulated = F * (1 + r)
    D_modulated -> DiffHead -> D

    重建过程（逆向）：
    C -> CommonDecoder -> C_recon
    D -> DiffDecoder -> D_recon
    [C_recon; D_recon] -> Fusion -> F_recon
    """

    def __init__(self, feat_dim: int, common_dim: int, diff_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.common_dim = common_dim
        self.diff_dim = diff_dim
        self.backbone_norm = nn.LayerNorm(feat_dim)

        # ========== 分解路径 ==========
        self.common_encoder = MLPBlock(feat_dim, common_dim, common_dim)
        self.common_aligner = nn.Sequential(
            nn.Linear(common_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )
        self.common_gate = nn.Sequential(
            nn.Linear(common_dim, feat_dim),
            nn.Sigmoid(),
        )
        self.residual_learner = ResidualLearner(feat_dim * 2, feat_dim)
        self.diff_input_norm = nn.LayerNorm(feat_dim)
        self.diff_head = MLPBlock(feat_dim, diff_dim, diff_dim)

        # ========== 重建路径（逆向） ==========
        self.common_decoder = MLPBlock(common_dim, common_dim, feat_dim)
        self.diff_decoder = MLPBlock(diff_dim, diff_dim, feat_dim)
        self.fusion = MLPBlock(feat_dim * 2, feat_dim, feat_dim)

    def decompose(
        self, backbone_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """分解过程。"""
        backbone_feat = self.backbone_norm(backbone_feat)
        c = self.common_encoder(backbone_feat)  # [B, common_dim]
        c_aligned = self.common_aligner(c)  # [B, feat_dim]
        gated_common = self.common_gate(c) * c_aligned
        diff_input = backbone_feat - gated_common
        combined = torch.cat([diff_input, c_aligned], dim=-1)  # [B, feat_dim * 2]
        r = self.residual_learner(combined)  # [B, feat_dim]
        d_modulated = diff_input * (1 + r)  # [B, feat_dim]
        d = self.diff_head(self.diff_input_norm(d_modulated))  # [B, diff_dim]
        return c, d, r, d_modulated, diff_input

    def reconstruct(self, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """重建过程（逆向）。"""
        c_recon = self.common_decoder(c)  # [B, feat_dim]
        d_recon = self.diff_decoder(d)  # [B, feat_dim]
        combined = torch.cat([c_recon, d_recon], dim=-1)  # [B, feat_dim * 2]
        f_recon = self.fusion(combined)  # [B, feat_dim]
        return f_recon

    def forward(self, backbone_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        """完整前向过程。"""
        c, d, r, d_modulated, diff_input = self.decompose(backbone_feat)
        f_recon = self.reconstruct(c, d)
        c_aligned = self.common_aligner(c)

        return {
            "common": c,
            "diff": d,
            "residual": r,
            "modulated": d_modulated,
            "diff_input": diff_input,
            "reconstructed": f_recon,
            "common_aligned": c_aligned,
        }


class LearnableResidualLoss(nn.Module):
    """
    学习式残差模块的损失函数。

    包含：
    1. 重建损失：确保信息不丢失
    2. 差模判别性：差模特征能区分身份
    3. 共模一致性：不同人的共模特征相似
    4. 反坍塌：防止共模特征退化为常数
    5. 共模反身份：共模特征不应包含身份信息（当前默认关闭）
    6. 调制因子正则：防止调制因子过大
    """

    def __init__(self, lambda_dict: dict[str, float] | None = None):
        super().__init__()
        self.lambda_dict = lambda_dict or {
            "reconstruction": 1.0,
            "diff_discrimination": 1.0,
            "common_consistency": 0.5,
            "anti_collapse": 0.1,
            "common_anti_identity": 0.1,
            "residual_reg": 0.01,
        }

    def forward(
        self,
        output: dict[str, torch.Tensor],
        backbone_feat: torch.Tensor,
        labels: torch.Tensor,
        pairs: torch.Tensor | None = None,
        classifier: nn.Module | None = None,
        common_classifier: nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses: dict[str, torch.Tensor] = {}

        # 1) 重建损失
        losses["reconstruction"] = F.mse_loss(output["reconstructed"], backbone_feat)

        # 2) 差模判别性
        if classifier is not None:
            logits = classifier(output["diff"])
            losses["diff_discrimination"] = F.cross_entropy(logits, labels)
        else:
            losses["diff_discrimination"] = torch.tensor(
                0.0, device=backbone_feat.device
            )

        # 3) 共模一致性：不同身份配对应共享更多共模
        if pairs is not None and len(pairs) > 0:
            c_i = output["common"][pairs[:, 0]]
            c_j = output["common"][pairs[:, 1]]
            losses["common_consistency"] = F.mse_loss(c_i, c_j)
        else:
            losses["common_consistency"] = torch.tensor(
                0.0, device=backbone_feat.device
            )

        # 4) 反坍塌：最大化共模方差（loss 取负）
        common_std = torch.std(output["common"], dim=0, unbiased=False).mean()
        losses["anti_collapse"] = -torch.clamp(common_std, min=1e-6)

        # 5) 共模反身份：默认关闭，保留接口便于后续打开
        _ = common_classifier
        losses["common_anti_identity"] = torch.tensor(0.0, device=backbone_feat.device)

        # 6) 调制因子正则
        losses["residual_reg"] = torch.mean(
            torch.clamp(torch.abs(output["residual"]), max=10.0)
        )

        total_loss = sum(self.lambda_dict.get(k, 1.0) * v for k, v in losses.items())
        return total_loss, losses


class DiffAuthModel(nn.Module):
    """DiffAuth 完整模型（骨干 + 分解模块 + 分类器）。"""

    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        common_dim: int,
        diff_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.residual_module = LearnableResidualModule(feat_dim, common_dim, diff_dim)

        # 差模分类器（用于身份识别）
        self.classifier = nn.Sequential(
            nn.Linear(diff_dim, diff_dim),
            nn.ReLU(),
            nn.Linear(diff_dim, num_classes),
        )

        # 共模分类器（用于对抗训练接口，默认未启用）
        self.common_classifier = nn.Sequential(
            nn.Linear(common_dim, common_dim),
            nn.ReLU(),
            nn.Linear(common_dim, num_classes),
        )

        self.feat_dim = feat_dim
        self.common_dim = common_dim
        self.diff_dim = diff_dim

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict[str, torch.Tensor]:
        """前向传播。"""
        _ = return_features
        backbone_feat = self.backbone(x)
        output = self.residual_module(backbone_feat)
        logits = self.classifier(output["diff"])
        return {"logits": logits, "backbone_feat": backbone_feat, **output}

    def get_common_features(self, x: torch.Tensor) -> torch.Tensor:
        backbone_feat = self.backbone(x)
        output = self.residual_module(backbone_feat)
        return output["common"]

    def get_diff_features(self, x: torch.Tensor) -> torch.Tensor:
        backbone_feat = self.backbone(x)
        output = self.residual_module(backbone_feat)
        return output["diff"]

    def classify(self, diff_features: torch.Tensor) -> torch.Tensor:
        return self.classifier(diff_features)


def create_pair_indices(batch_size: int, labels: torch.Tensor) -> torch.Tensor | None:
    """
    创建相似条件样本对索引。

    当前策略：将同一 batch 里“标签不同”的样本做配对，
    以约束共模特征在身份上更一致。
    """
    pairs: list[list[int]] = []
    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels

    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if labels_np[i] != labels_np[j]:
                pairs.append([i, j])

    if not pairs:
        return None
    return torch.tensor(pairs, dtype=torch.long, device=labels.device)


def compute_metrics(output: dict[str, torch.Tensor], labels: torch.Tensor) -> dict[str, float]:
    """计算基础评估指标。"""
    with torch.no_grad():
        preds = torch.argmax(output["logits"], dim=1)
        accuracy = (preds == labels).float().mean()
        common_std = torch.std(output["common"], dim=0, unbiased=False).mean()
        diff_std = torch.std(output["diff"], dim=0, unbiased=False).mean()
        residual_abs = torch.abs(output["residual"])
        residual_mean = residual_abs.mean()
        residual_max = residual_abs.max()

    return {
        "accuracy": float(accuracy.item()),
        "common_std": float(common_std.item()),
        "diff_std": float(diff_std.item()),
        "residual_mean": float(residual_mean.item()),
        "residual_max": float(residual_max.item()),
    }


if __name__ == "__main__":
    # 简单自检
    batch_size = 32
    feat_dim = 256
    common_dim = 128
    diff_dim = 128
    num_classes = 10

    module = LearnableResidualModule(feat_dim, common_dim, diff_dim)
    backbone_feat = torch.randn(batch_size, feat_dim)
    output = module(backbone_feat)

    print("=== Output Shapes ===")
    print(f"Common features: {output['common'].shape}")
    print(f"Diff features: {output['diff'].shape}")
    print(f"Residual: {output['residual'].shape}")
    print(f"Modulated features: {output['modulated'].shape}")
    print(f"Reconstructed features: {output['reconstructed'].shape}")

    labels = torch.randint(0, num_classes, (batch_size,))
    classifier = nn.Linear(diff_dim, num_classes)
    common_classifier = nn.Linear(common_dim, num_classes)

    loss_fn = LearnableResidualLoss()
    total_loss, losses = loss_fn(
        output,
        backbone_feat,
        labels,
        classifier=classifier,
        common_classifier=common_classifier,
    )

    print("\n=== Loss Values ===")
    for k, v in losses.items():
        print(f"{k}: {v.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")

import numpy as np
import torch
import matplotlib.pyplot as plt
from data_loader import load_time_series

# Explicit parameters (no CLI)
DATA_PATH = "data/train/data_time_domain.txt"
TARGET_LABEL = "shake"
CORRELATION_THRESHOLD = 0.85
BLOCK_SIZE = 30
MAX_SAMPLES = None  # e.g. 900
DEVICE = "cuda"
OUTPUT_PATH = "template_subset.txt"
CDF_PLOT_PATH = "template_coverage_cdf.png"
TEMPLATE_PLOT_PATH = "selected_templates.png"
MAX_TEMPLATE_PLOTS = 12
MAX_CORR_DIST_PLOT_PATH = "max_correlation_to_templates.png"
HIST_BINS = 50


def _zscore(x, dim=-1, eps=1e-8):
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True) + eps
    return (x - mean) / std


def max_circular_correlation_block(x_block, y_block):
    """
    Compute max circular correlation between two blocks.

    Args:
        x_block: (Bx, C, L) float tensor
        y_block: (By, C, L) float tensor
    Returns:
        (Bx, By) tensor of average max correlation across channels
    """
    _, channels, length = x_block.shape

    x_norm = _zscore(x_block, dim=-1)
    y_norm = _zscore(y_block, dim=-1)

    x_fft = torch.fft.fft(x_norm, dim=-1)  # (Bx, C, L)
    y_fft = torch.fft.fft(y_norm, dim=-1)  # (By, C, L)

    # Broadcast to pairwise products: (Bx, By, C, L)
    cross = x_fft[:, None, :, :] * torch.conj(y_fft)[None, :, :, :]
    corr = torch.fft.ifft(cross, dim=-1).real / length

    # Max over shifts, then average over channels
    max_corr = corr.max(dim=-1).values  # (Bx, By, C)
    avg_corr = max_corr.mean(dim=2)  # (Bx, By)
    return avg_corr


def build_coverage_matrix(data, threshold=0.8, block_size=30, device="cpu"):
    """
    Build boolean coverage matrix where cover[i, j] = True if sample i covers j.
    """
    n = data.shape[0]
    cover = np.zeros((n, n), dtype=bool)

    data_t = torch.from_numpy(data).to(device)

    for i in range(0, n, block_size):
        x_block = data_t[i : i + block_size]
        for j in range(0, n, block_size):
            y_block = data_t[j : j + block_size]
            with torch.no_grad():
                sim = max_circular_correlation_block(x_block, y_block)
            cover_block = (sim >= threshold).cpu().numpy()
            cover[i : i + cover_block.shape[0], j : j + cover_block.shape[1]] = cover_block

    return cover


def greedy_min_cover(cover):
    """
    Greedy set cover for boolean coverage matrix.
    """
    n = cover.shape[0]
    uncovered = np.ones(n, dtype=bool)
    selected = []

    while uncovered.any():
        # Count how many uncovered samples each candidate can cover
        counts = cover[:, uncovered].sum(axis=1)
        best = int(np.argmax(counts))
        if counts[best] == 0:
            break
        selected.append(best)
        uncovered[cover[best]] = False

    return selected


def plot_correlation_distribution(values, title, output_path):
    if values.size == 0:
        print(f"Skipped plot (no values) for {output_path}")
        return

    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(values, bins=HIST_BINS, alpha=0.8)
    plt.title(f"{title} - Histogram")
    plt.xlabel("Max Circular Correlation")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(sorted_vals, cdf, linewidth=2)
    plt.title(f"{title} - CDF")
    plt.xlabel("Max Circular Correlation")
    plt.ylabel("Cumulative Probability")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Correlation plot saved to {output_path}")


def compute_max_corr_to_templates(data, selected, block_size=30, device="cpu"):
    """
    For each sample, compute max correlation to any selected template.
    """
    if len(selected) == 0:
        return np.array([])

    data_t = torch.from_numpy(data).to(device)
    templates_t = data_t[selected]
    max_corr = None

    for i in range(0, templates_t.shape[0], block_size):
        x_block = templates_t[i : i + block_size]
        block_max = []
        for j in range(0, data_t.shape[0], block_size):
            y_block = data_t[j : j + block_size]
            with torch.no_grad():
                sim = max_circular_correlation_block(x_block, y_block)  # (Bx, By)
            block_max.append(sim.max(dim=0).values.cpu().numpy())  # (By,)

        block_max = np.concatenate(block_max)
        if max_corr is None:
            max_corr = block_max
        else:
            max_corr = np.maximum(max_corr, block_max)

    return max_corr


def plot_selected_templates(data, selected, output_path):
    if len(selected) == 0:
        print("Template plot skipped (no selected templates).")
        return

    count = min(len(selected), MAX_TEMPLATE_PLOTS)
    cols = 2
    rows = count
    plt.figure(figsize=(12, 2.5 * rows))

    for i in range(count):
        idx = selected[i]
        accel = data[idx, 0, :]
        gyro = data[idx, 1, :]

        ax1 = plt.subplot(rows, cols, i * 2 + 1)
        ax1.plot(accel, linewidth=1.2)
        ax1.set_title(f"Template {i + 1} - Accel (idx={idx})")
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(rows, cols, i * 2 + 2)
        ax2.plot(gyro, linewidth=1.2, color="orange")
        ax2.set_title(f"Template {i + 1} - Gyro (idx={idx})")
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Template plot saved to {output_path}")


def main():
    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    data, labels, users, file_ids = load_time_series(DATA_PATH, target_label=TARGET_LABEL)

    if MAX_SAMPLES is not None and len(data) > MAX_SAMPLES:
        data = data[:MAX_SAMPLES]
        labels = labels[:MAX_SAMPLES]
        users = users[:MAX_SAMPLES]
        file_ids = file_ids[:MAX_SAMPLES]

    cover = build_coverage_matrix(
        data, threshold=CORRELATION_THRESHOLD, block_size=BLOCK_SIZE, device=device
    )
    selected = greedy_min_cover(cover)

    covered_mask = np.zeros(len(data), dtype=bool)
    for idx in selected:
        covered_mask |= cover[idx]

    selected_cover_counts = np.array([int(cover[idx].sum()) for idx in selected])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("index\tfile_id\tuser\tlabel\tcovered_count\n")
        for idx in selected:
            f.write(
                f"{idx}\t{file_ids[idx]}\t{users[idx]}\t{labels[idx]}\t{int(cover[idx].sum())}\n"
            )

    coverage_rate = covered_mask.mean() * 100.0
    print(f"Selected {len(selected)} templates out of {len(data)} samples.")
    print(f"Coverage: {coverage_rate:.2f}%")
    print(f"Saved to {OUTPUT_PATH}")

    # Cumulative coverage curve: sort templates by coverage count (desc),
    # then plot cumulative covered fraction as templates are added.
    if len(selected) > 0:
        order = np.argsort(-selected_cover_counts)
        cumulative_cover = []
        covered = np.zeros(len(data), dtype=bool)
        for idx in order:
            covered |= cover[selected[idx]]
            cumulative_cover.append(covered.mean())

        x = np.arange(1, len(cumulative_cover) + 1)
        y = np.array(cumulative_cover) * 100.0
        plt.figure(figsize=(8, 5))
        plt.step(x, y, where="post", linewidth=2)
        plt.title("Cumulative Coverage vs Template Count (Sorted by Coverage)")
        plt.xlabel("Number of Templates")
        plt.ylabel("Covered Samples (%)")
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(CDF_PLOT_PATH, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"CDF plot saved to {CDF_PLOT_PATH}")
    else:
        print("CDF plot skipped (no selected templates).")

    # Visualize selected templates
    plot_selected_templates(data, selected, TEMPLATE_PLOT_PATH)

    # Distribution of max correlation to templates (per sample)
    max_corr = compute_max_corr_to_templates(data, selected, block_size=BLOCK_SIZE, device=device)
    plot_correlation_distribution(
        max_corr, "Max Correlation to Templates (Per Sample)", MAX_CORR_DIST_PLOT_PATH
    )


if __name__ == "__main__":
    main()

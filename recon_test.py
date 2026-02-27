import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from data_loader import get_data_loader, load_time_series
from recon_model import ReconModel, compute_reconstruction_loss

# Explicit parameters (no CLI)
TRAIN_DATA_PATH = "data/train/data_time_domain.txt"
TEST_DATA_PATH = "data/test/data_time_domain.txt"
TEMPLATE_INDEX_PATH = "template_subset.txt"
RECON_THRESHOLD = 0.1
CORRELATION_THRESHOLD = 0.8
BLOCK_SIZE = 30
NON_TARGET_PLOT_DIR = "non_target_accepted_plots"


def _zscore(x, dim=-1, eps=1e-8):
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True) + eps
    return (x - mean) / std


def max_circular_correlation_block(x_block, y_block):
    """
    Compute max circular correlation between two blocks.
    Args:
        x_block: (Bx, C, L)
        y_block: (By, C, L)
    Returns:
        (Bx, By) average max correlation across channels.
    """
    _, channels, length = x_block.shape
    x_norm = _zscore(x_block, dim=-1)
    y_norm = _zscore(y_block, dim=-1)

    x_fft = torch.fft.fft(x_norm, dim=-1)
    y_fft = torch.fft.fft(y_norm, dim=-1)
    cross = x_fft[:, None, :, :] * torch.conj(y_fft)[None, :, :, :]
    corr = torch.fft.ifft(cross, dim=-1).real / length
    max_corr = corr.max(dim=-1).values  # (Bx, By, C)
    return max_corr.mean(dim=2)


def load_templates(train_path, template_index_path, device):
    data, _, _, _ = load_time_series(train_path, target_label="shake")
    with open(template_index_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()

    if len(lines) <= 1:
        raise ValueError("Template index file is empty.")

    indices = []
    for line in lines[1:]:
        parts = line.split("\t")
        if parts:
            indices.append(int(parts[0]))

    templates = torch.from_numpy(data[indices]).to(device)
    return templates


def max_correlation_to_templates(samples, templates, block_size=30):
    """
    For each sample, compute max correlation to any template.
    Args:
        samples: (N, C, L) tensor
        templates: (T, C, L) tensor
    Returns:
        (N,) tensor
    """
    if templates.numel() == 0:
        return torch.zeros(samples.shape[0], device=samples.device)

    max_corr = None
    for i in range(0, templates.shape[0], block_size):
        t_block = templates[i : i + block_size]
        block_max = []
        for j in range(0, samples.shape[0], block_size):
            s_block = samples[j : j + block_size]
            sim = max_circular_correlation_block(t_block, s_block)
            block_max.append(sim.max(dim=0).values)
        block_max = torch.cat(block_max)
        max_corr = block_max if max_corr is None else torch.maximum(max_corr, block_max)
    return max_corr


def find_best_template(sample, templates, block_size=30):
    """
    Find the best-matching template for a single sample.
    Args:
        sample: (C, L) tensor
        templates: (T, C, L) tensor
    Returns:
        (best_index, best_corr)
    """
    if templates.numel() == 0:
        return -1, 0.0

    best_idx = -1
    best_corr = -1.0
    sample = sample.unsqueeze(0)  # (1, C, L)

    for i in range(0, templates.shape[0], block_size):
        t_block = templates[i : i + block_size]
        sim = max_circular_correlation_block(t_block, sample)  # (B, 1)
        block_max, block_idx = sim.max(dim=0)
        corr_val = float(block_max.item())
        if corr_val > best_corr:
            best_corr = corr_val
            best_idx = i + int(block_idx.item())

    return best_idx, best_corr


def _safe_name(value):
    name = str(value)
    name = name.replace(os.sep, "_").replace(" ", "_")
    return name


def plot_non_target_match(sample, template, label, file_id, corr_value, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sample_np = sample.detach().cpu().numpy()
    template_np = template.detach().cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(sample_np[0], linewidth=1.2)
    plt.title("Sample - Accel")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(template_np[0], linewidth=1.2, color="green")
    plt.title("Best Template - Accel")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(sample_np[1], linewidth=1.2, color="orange")
    plt.title("Sample - Gyro")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(template_np[1], linewidth=1.2, color="red")
    plt.title("Best Template - Gyro")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    plt.suptitle(f"Label={label} | Max Corr={corr_value:.4f}", fontsize=12)
    plt.tight_layout()
    fname = f"{_safe_name(label)}_{_safe_name(file_id)}_corr_{corr_value:.4f}.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
    plt.close()


def visualize_reconstruction_samples(model, loader, num_samples=5):
    """Visualize reconstruction samples for a single sample across all channels"""
    # Get device from model
    device = next(model.parameters()).device
    print(f"Visualizing on device: {device}")
    
    model.eval()
    with torch.no_grad():
        # Get a batch of data
        for batch in loader:
            time_data = batch['time_data'].to(device)
            labels = batch['raw_label']
            break
        
        # Process the data through the model
        output = model(time_data)
        
        # Visualize only the first sample across all channels
        sample_idx = 0
        num_channels = time_data.shape[1]
        
        # Create figure for reconstruction visualization
        # If no frequency data, only show time domain
        fig, axes = plt.subplots(num_channels, 1, figsize=(15, 3*num_channels))
        fig.suptitle(f'Original vs Reconstructed Signals - Sample {sample_idx} (Label: {labels[sample_idx]})', fontsize=16)
        
        for ch in range(num_channels):
            axes[ch].plot(time_data[sample_idx, ch, :].cpu().numpy(), label='Original', alpha=0.7)
            axes[ch].plot(output['time_reconstructed'][sample_idx, ch, :].cpu().numpy(), label='Reconstructed', alpha=0.7)
            axes[ch].set_title(f'Time Domain - Channel {ch}')
            axes[ch].set_xlabel('Time Steps')
            axes[ch].set_ylabel('Amplitude')
            axes[ch].legend()
            axes[ch].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reconstruction_samples.png', dpi=300, bbox_inches='tight')
        plt.show()

def visualize_reconstruction_loss_distribution(model, loader):
    """Visualize reconstruction loss distribution"""
    # Get device from model
    device = next(model.parameters()).device
    print(f"Visualizing on device: {device}")
    
    model.eval()
    time_recon_losses = []
    freq_recon_losses = []
    labels_list = []
    
    with torch.no_grad():
        for batch in loader:
            time_data = batch['time_data'].to(device)
            labels = batch['raw_label']
            
            output = model(time_data)
            
            # Compute reconstruction losses
            time_recon_loss = compute_reconstruction_loss(time_data, output['time_reconstructed'])
            time_recon_losses.extend(time_recon_loss.cpu().numpy())
            
            labels_list.extend(labels)
    
    # Convert to numpy arrays
    time_recon_losses = np.array(time_recon_losses)
    labels_array = np.array(labels_list)
    
    # Create figure for loss distribution
    fig, axes = plt.subplots(1, 1, figsize=(15, 6))
    axes = [axes]
    
    # Plot time domain reconstruction loss distribution
    unique_labels = np.unique(labels_array)
    for label in unique_labels:
        mask = labels_array == label
        axes[0].hist(time_recon_losses[mask], alpha=0.7, label=f'{label} (n={np.sum(mask)})', bins=50)
    
    axes[0].set_title('Time Domain Reconstruction Loss Distribution')
    axes[0].set_xlabel('Reconstruction Loss')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot frequency domain reconstruction loss distribution (if available)
    plt.tight_layout()
    plt.savefig('reconstruction_loss_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\nReconstruction Loss Statistics:")
    print("-" * 40)
    print(f"Time Domain - Mean: {np.mean(time_recon_losses):.6f}, Std: {np.std(time_recon_losses):.6f}")
    # Frequency domain disabled


def test_separate_domain_model(time_domain_file=None):
    """Test the time-domain authentication model with different OOD detection methods"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Testing Separate Domain Authentication System")
    print("=" * 50)
    print(f"Reconstruction Threshold: {RECON_THRESHOLD}")
    print(f"Template Matching Correlation Threshold: {CORRELATION_THRESHOLD}")

    print()
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the trained model
    try:
        # Try to load the best model first
        checkpoint_path = os.path.join(current_dir, 'recon_model_best.pth')
        if not os.path.exists(checkpoint_path):
            # Fall back to the regular model if best model doesn't exist
            checkpoint_path = os.path.join(current_dir, 'recon_model.pth')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = ReconModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        model.eval()
        print(f"Model loaded successfully from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create data loader for visualization
    if time_domain_file is None:
        time_domain_file = os.path.join(current_dir, TEST_DATA_PATH)

    loader = get_data_loader(
        time_domain_file,
        batch_size=32,
        shuffle=False
    )
    
    # Print mode information
    print("Mode: Time Domain Only")

    # Load templates (from train data using selected indices)
    templates = load_templates(
        os.path.join(current_dir, TRAIN_DATA_PATH),
        os.path.join(current_dir, TEMPLATE_INDEX_PATH),
        device
    )
    
    # Visualize reconstruction samples
    print("\nGenerating reconstruction samples visualization...")
    try:
        visualize_reconstruction_samples(model, loader)
        print("Reconstruction samples visualization saved as 'reconstruction_samples.png'")
    except Exception as e:
        print(f"Error generating reconstruction samples visualization: {e}")
    
    # Visualize reconstruction loss distribution
    print("\nGenerating reconstruction loss distribution visualization...")
    try:
        visualize_reconstruction_loss_distribution(model, loader)
        print("Reconstruction loss distribution visualization saved as 'reconstruction_loss_distribution.png'")
    except Exception as e:
        print(f"Error generating reconstruction loss distribution visualization: {e}")
    
    # Test: Overall authentication statistics
    print("\nTest : Overall Authentication Statistics")
    print("-" * 40)
    
    total_samples = 0
    accepted_samples = 0
    shake_samples = 0
    shake_accepted = 0
    non_shake_samples = 0
    non_shake_accepted = 0
    label_counts = {}
    label_accepted = {}
    
    with torch.no_grad():
        for batch in loader:
            time_data = batch['time_data'].to(device)
            labels = batch['raw_label']
            users = batch['user']
            file_ids = batch['file_id']
            
            output = model(time_data)
            
            time_recon_loss = compute_reconstruction_loss(time_data, output['time_reconstructed'])
            max_corr = max_correlation_to_templates(
                time_data, templates, block_size=BLOCK_SIZE
            )
            authenticated = (time_recon_loss < RECON_THRESHOLD) & (max_corr >= CORRELATION_THRESHOLD)
            
            total_samples += len(labels)
            accepted_samples += authenticated.sum().item()

            # Per-label stats
            for i, label in enumerate(labels):
                label_counts[label] = label_counts.get(label, 0) + 1
                label_accepted[label] = label_accepted.get(label, 0) + int(authenticated[i].item())
            
            # Count shake samples
            shake_mask = torch.tensor(np.array(labels) == 'shake', device=device)
            shake_samples += shake_mask.sum().item()
            shake_accepted += (authenticated & shake_mask).sum().item()
            
            # Count non-shake samples (all should be rejected)
            non_shake_mask = torch.tensor(np.array(labels) != 'shake', device=device)
            non_shake_samples += non_shake_mask.sum().item()
            non_shake_accepted += (authenticated & non_shake_mask).sum().item()

            # Save plots for accepted non-target samples
            for i, label in enumerate(labels):
                if label != 'shake' and authenticated[i].item():
                    best_idx, best_corr = find_best_template(
                        time_data[i], templates, block_size=BLOCK_SIZE
                    )
                    if best_idx >= 0:
                        plot_non_target_match(
                            time_data[i],
                            templates[best_idx],
                            label,
                            file_ids[i],
                            best_corr,
                            NON_TARGET_PLOT_DIR
                        )
    
    overall_acceptance_rate = accepted_samples / total_samples * 100
    shake_acceptance_rate = shake_accepted / shake_samples * 100 if shake_samples > 0 else 0
    non_shake_acceptance_rate = non_shake_accepted / non_shake_samples * 100 if non_shake_samples > 0 else 0
    
    print(f"Overall Results:")
    print(f"Total samples: {total_samples}")
    print(f"Accepted samples: {accepted_samples}")
    print(f"Overall acceptance rate: {overall_acceptance_rate:.2f}%")
    print(f"Overall rejection rate: {100 - overall_acceptance_rate:.2f}%")
    
    print(f"\nDetailed Results by Label:")
    print(f"'shake' samples: {shake_samples}")
    print(f"'shake' accepted: {shake_accepted}")
    print(f"'shake' acceptance rate: {shake_acceptance_rate:.2f}%")
    print(f"'shake' rejection rate: {100 - shake_acceptance_rate:.2f}%")
    
    print(f"\n'non-shake' samples: {non_shake_samples}")
    print(f"'non-shake' accepted: {non_shake_accepted}")
    print(f"'non-shake' acceptance rate: {non_shake_acceptance_rate:.2f}%")
    print(f"'non-shake' rejection rate: {100 - non_shake_acceptance_rate:.2f}%")

    print("\nPer-Label Breakdown:")
    print("-" * 40)
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        accepted = label_accepted.get(label, 0)
        acc_rate = accepted / count * 100 if count > 0 else 0
        rej_rate = 100 - acc_rate
        print(f"{label}: {accepted}/{count} accepted ({acc_rate:.2f}%), rejected {rej_rate:.2f}%")
    
    print("\nSystem Features:")
    print("- Separate time and frequency domain processing")
    print("- Robust OOD (Out-of-Distribution) rejection")
    print("- Supports time-domain, frequency-domain, or both")


if __name__ == "__main__":
    import os
    # Test with time domain data only
    print("Testing with Time Domain Data Only:")
    test_separate_domain_model()

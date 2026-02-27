import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_loader import get_data_loader
from recon_model import ReconModel
import os


def train_model(time_domain_file, target_label='shake',
                epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train the separate domain authentication model with three-stage template learning

    Args:
        time_domain_file (str): Path to time domain CSV file
        target_label (str): Target action label for authentication
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    # Get the directory of the current script for saving models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create data loader (train file only)
    num_workers = max(0, (os.cpu_count() or 4) // 2)
    train_loader = get_data_loader(
        time_domain_file,
        target_label=target_label,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0)
    )
    
    # Initialize model
    model = ReconModel()
    
    # Move model to device
    model = model.to(device)
    
    # Train reconstruction model
    print("\n=== Training Reconstruction Model ===")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_recon_loss = 0
        num_batches = 0

        for batch in train_loader:
            time_data = batch['time_data'].to(device)
            output = model(time_data)
            recon_loss = F.mse_loss(output['time_reconstructed'], time_data)

            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            num_batches += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Recon Loss: {total_recon_loss/num_batches:.6f}")
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'time_input_channels': 2
    }, os.path.join(current_dir, 'recon_model_best.pth'))
    
    print("\nTraining completed. Model saved as 'recon_model_best.pth'")
    
    return model


# Removed validate_model_split function as per user request to remove all validation logic


if __name__ == "__main__":
    import os
    # Example usage
    print("Training separate domain authentication model...")
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Train the model with data split
    model = train_model(
        time_domain_file=os.path.join(current_dir, 'data/train/data_time_domain.txt'),
        target_label='shake',
        epochs=200,
        batch_size=32,
        learning_rate=0.001
    )
    
    print("\nTraining completed. Best model already saved during training.")

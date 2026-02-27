import torch
import torch.nn as nn


class ReconModel(nn.Module):
    """Simple time-domain autoencoder (2-channel input)."""
    def __init__(self):
        super().__init__()
        self.time_conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.time_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.time_conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        self.time_deconv3 = nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1)
        self.time_deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1)
        self.time_deconv1 = nn.ConvTranspose1d(32, 2, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(2, return_indices=True)
        self.unpool = nn.MaxUnpool1d(2)
        self.relu = nn.ReLU()

    def forward(self, time_data):
        x1 = self.relu(self.time_conv1(time_data))
        x1_pooled, idx1 = self.pool(x1)

        x2 = self.relu(self.time_conv2(x1_pooled))
        x2_pooled, idx2 = self.pool(x2)

        x3 = self.relu(self.time_conv3(x2_pooled))
        x3_pooled, idx3 = self.pool(x3)

        x3_up = self.unpool(x3_pooled, idx3, output_size=x3.size())
        x3_up = self.relu(self.time_deconv3(x3_up))

        x2_up = self.unpool(x3_up, idx2, output_size=x2.size())
        x2_up = self.relu(self.time_deconv2(x2_up))

        x1_up = self.unpool(x2_up, idx1, output_size=x1.size())
        time_reconstructed = self.time_deconv1(x1_up)

        return {"time_reconstructed": time_reconstructed}


def compute_reconstruction_loss(original_data, reconstructed_data):
    if original_data.device != reconstructed_data.device:
        reconstructed_data = reconstructed_data.to(original_data.device)
    return torch.mean((original_data - reconstructed_data) ** 2, dim=(1, 2))

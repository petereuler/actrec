import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def _read_table(path):
    """Read CSV or TXT with auto delimiter detection."""
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in {".csv", ".txt"}:
        # sep=None triggers python engine delimiter sniffing (comma, space, tab, etc.)
        return pd.read_csv(path, sep=None, engine="python")
    # Fallback: try as csv with delimiter sniffing
    return pd.read_csv(path, sep=None, engine="python")


def load_time_series(time_domain_file, target_label=None):
    """
    Load time-domain IMU data as combined accel/gyro channels.

    Returns:
        tuple: (time_series, labels, users, file_ids)
            time_series: np.ndarray (N, 2, 128)
    """
    data = _read_table(time_domain_file)
    if target_label is not None:
        data = data[data["label"] == target_label].reset_index(drop=True)

    sensor_columns = [col for col in data.columns if col.startswith("mobile_")]
    if not sensor_columns:
        raise ValueError("No sensor columns found (expected columns starting with 'mobile_').")

    raw = data[sensor_columns].values.astype(np.float32)
    if raw.shape[1] != 6 * 128:
        raise ValueError(f"Expected 768 sensor columns (6x128), got {raw.shape[1]}.")

    raw = raw.reshape(-1, 128, 6).transpose(0, 2, 1)  # (N, 6, 128)
    combined_accel = np.sqrt(raw[:, 0, :] ** 2 + raw[:, 1, :] ** 2 + raw[:, 2, :] ** 2)
    combined_gyro = np.sqrt(raw[:, 3, :] ** 2 + raw[:, 4, :] ** 2 + raw[:, 5, :] ** 2)
    time_series = np.stack([combined_accel, combined_gyro], axis=1)  # (N, 2, 128)

    labels = data["label"].values
    users = data["user"].values
    if "file_id" in data.columns:
        file_ids = data["file_id"].values
    else:
        file_ids = np.array([f"sample_{i}" for i in range(len(data))])

    return time_series, labels, users, file_ids


class IMUDataset(Dataset):
    """Dataset class for IMU data loading (time domain only)"""
    def __init__(self, time_domain_file, target_label=None, target_user=None, indices=None):
        """
        Args:
            time_domain_file (str): Path to time domain CSV file
            frequency_domain_file (str, optional): Path to frequency domain CSV file
            target_label (str, optional): Target action label for authentication
            target_user (str, optional): Target user ID for authentication
            indices (array-like, optional): Indices to select specific samples
        """
        self.time_data = _read_table(time_domain_file)
        
        if indices is not None:
            self.time_data = self.time_data.iloc[indices].reset_index(drop=True)
            
        self.target_label = target_label
        self.target_user = target_user
        
        # Define sensor data columns (excluding metadata)
        self.sensor_columns = [col for col in self.time_data.columns if col.startswith('mobile_')]
        self.meta_columns = ['label', 'sub_label', 'user', 'file_id']

        if not self.sensor_columns:
            raise ValueError("No sensor columns found (expected columns starting with 'mobile_').")

        raw = self.time_data[self.sensor_columns].values.astype(np.float32)
        if raw.shape[1] != 6 * 128:
            raise ValueError(f"Expected 768 sensor columns (6x128), got {raw.shape[1]}.")

        raw = raw.reshape(-1, 128, 6).transpose(0, 2, 1)  # (N, 6, 128)
        combined_accel = np.sqrt(
            raw[:, 0, :] ** 2 +
            raw[:, 1, :] ** 2 +
            raw[:, 2, :] ** 2
        )
        combined_gyro = np.sqrt(
            raw[:, 3, :] ** 2 +
            raw[:, 4, :] ** 2 +
            raw[:, 5, :] ** 2
        )
        self.time_series = torch.from_numpy(
            np.stack([combined_accel, combined_gyro], axis=1)
        )  # (N, 2, 128)
        self.labels = self.time_data['label'].values
        self.users = self.time_data['user'].values
        if 'file_id' in self.time_data.columns:
            self.file_ids = self.time_data['file_id'].values
        else:
            self.file_ids = np.array([f'sample_{i}' for i in range(len(self.time_data))])
    
    def __len__(self):
        return len(self.time_data)
    
    def __getitem__(self, idx):
        # Extract metadata
        time_series = self.time_series[idx]
        label = self.labels[idx]
        user = self.users[idx]
        file_id = self.file_ids[idx]
        
        # Create label: 1 for target class, 0 for others
        # Modified to only consider target_label, ignoring target_user
        if self.target_label:
            class_label = 1 if label == self.target_label else 0
        else:
            class_label = 1  # If no target specified, treat all as positive (for training)
        
        result = {
            'time_data': time_series.float(),
            'label': torch.LongTensor([class_label]),
            'raw_label': label,
            'user': user,
            'file_id': file_id
        }
        
        return result


def get_data_loader(time_domain_file, target_label=None, target_user=None,
                   batch_size=32, shuffle=True, indices=None,
                   num_workers=0, pin_memory=None, persistent_workers=False):
    """Create a DataLoader for IMU data (time domain only)"""
    dataset = IMUDataset(time_domain_file, target_label, target_user, indices)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )


def get_data_loaders_split(time_domain_file, target_label=None, target_user=None,
                          train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, batch_size=32, random_state=42):
    """
    Create train, validation, and test DataLoaders with specified ratios.
    
    Args:
        time_domain_file (str): Path to time domain CSV file
        target_label (str, optional): Target action label for authentication
        target_user (str, optional): Target user ID for authentication
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        batch_size (int): Batch size for DataLoader
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Read data to get indices
    time_data = _read_table(time_domain_file)
    
    # Get indices for target label only
    if target_label:
        target_indices = time_data[time_data['label'] == target_label].index.values
    else:
        target_indices = time_data.index.values
    
    # Split indices
    train_indices, temp_indices = train_test_split(
        target_indices, train_size=train_ratio, random_state=random_state
    )
    
    # Calculate validation ratio from remaining data
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices, train_size=val_ratio_adjusted, random_state=random_state
    )
    
    # Create DataLoaders
    train_loader = get_data_loader(
        time_domain_file, target_label, target_user,
        batch_size=batch_size, shuffle=True, indices=train_indices
    )
    
    val_loader = get_data_loader(
        time_domain_file, target_label, target_user,
        batch_size=batch_size, shuffle=False, indices=val_indices
    )
    
    test_loader = get_data_loader(
        time_domain_file, target_label, target_user,
        batch_size=batch_size, shuffle=False, indices=test_indices
    )
    
    print(f"Data split - Train: {len(train_indices)}, Validation: {len(val_indices)}, Test: {len(test_indices)}")
    
    return train_loader, val_loader, test_loader


# Example usage:
# loader = get_data_loader('data/train/data_time_domain.txt',
#                         target_label='shake', target_user='10544325')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import onnx
from torch.nn.parallel import DataParallel
import matplotlib.pyplot as plt

import os
import argparse

import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from src.models.s4.s4 import S4
from src.models.s4.s4d import S4D
from tqdm.auto import tqdm
from Wavenet import WaveNetClassifier
import csv

from ncps import wirings
from ncps.torch import CfC, LTC

from scipy.signal import butter, filtfilt, resample
from scipy import signal

import pywt


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_fscore_support



# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--file_name', default='S4D_NCP_test1', type=str, help='Folder Name')
parser.add_argument('--n_channels', type=int, default=0, help='Number of channels are emptied')
# Optimizer
parser.add_argument('--lr', default=0.003, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
# Scheduler
# parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--epochs2', default=300, type=int, help='Training epochs')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
# Model
parser.add_argument('--n_layers', default=1, type=int, help='Number of layers')
parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')

parser.add_argument('--cuda', default=0, type=int, help='Cuda')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


d_input = 12
d_output = 6
n = args.n_channels
n_layers = args.n_layers
batch_size =  args.batch_size

# Define the directory path and file name where you want to save the text file
output_directory = '...' + args.file_name

if not os.path.exists(output_directory):
    # If it doesn't exist, create the directory
    os.makedirs(output_directory)
    print(f"Directory '{output_directory}' created successfully.")
else:
    print(f"Directory '{output_directory}' already exists.")



class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        wiring = wirings.AutoNCP(20, d_output)
        ncp = CfC(d_model, wiring, batch_first=True)  # , return_sequences=False

        self.decoder = ncp

        # self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)
        # print(x.shape)
        # Decode the outputs
        x , _ = self.decoder(x)  # (B, d_model) -> (B, d_output)
        # print(x.shape)
        return x

# Model
print('==> Building model..')
model = S4Model(
    d_input=d_input,
    d_output=d_output,
    d_model=args.d_model,
    n_layers=args.n_layers,
    dropout=args.dropout,
    prenorm=args.prenorm,
)
# Load the saved model from .pt file
state_dict = torch.load(output_directory + '/model_' + str(n_layers) + '_' + str(args.lr) + '-' + str(args.epochs2) + '.pt') #

# Load the state dictionary into the model
model.load_state_dict(state_dict)
model = model.to(device)


# Filters
def apply_bandpass_filter(ecg_data, fs=500, lowcut=0.5, highcut=40, order=4):
    """
    Applies a bandpass filter to each lead in ECG data.

    Args:
    ecg_data (numpy.ndarray): numpy array of shape [N, 4096, 12], where N is the number of ECG recordings
    fs (float): Sampling frequency in Hz (default: 500 Hz)
    lowcut (float): Lower cutoff frequency in Hz (default: 0.5 Hz)
    highcut (float): Upper cutoff frequency in Hz (default: 40 Hz)
    order (int): Filter order (default: 4)

    Returns:
    numpy.ndarray: a numpy array of shape [N, 4096, 12], containing the denoised ECG data
    """
    nyq = 0.5*fs
    lowcut = lowcut/nyq
    highcut = highcut/nyq

    # Create an empty array to store the denoised ECG data
    denoised_ecg_data = np.zeros_like(ecg_data)

    # Loop through each lead in the ECG data
    for i in range(ecg_data.shape[0]):
        for j in range(ecg_data.shape[2]):
            # Extract the ECG data for the current lead
            lead_data = ecg_data[i, :, j]

            # Design the bandpass filter
            b, a = butter(order,[lowcut,highcut], btype='band')

            # Apply the bandpass filter to the lead data
            denoised_lead_data = filtfilt(b, a, lead_data)

            # Store the denoised lead data in the denoised ECG data array
            denoised_ecg_data[i, :, j] = denoised_lead_data

    return denoised_ecg_data

def filter_ecg_signal(data, wavelet='db4', level=8, fs=500, fc=0.1, order=6):
    """
    Filter ECG signals using wavelet denoising.

    Args:
        data (numpy array): ECG signal data with shape (n_samples, n_samples_per_lead, n_leads).
        wavelet (str, optional): Wavelet type for denoising. Default is 'db4'.
        level (int, optional): Decomposition level for wavelet denoising. Default is 8.
        fs (float, optional): Sampling frequency of ECG signals. Default is 500 Hz.
        fc (float, optional): Cutoff frequency for lowpass filter. Default is 0.1 Hz.
        order (int, optional): Filter order for Butterworth filter. Default is 6.

    Returns:
        numpy array: Filtered ECG signals.
    """
    nyquist = 0.5 * fs
    cutoff = fc / nyquist
    b, a = signal.butter(order, cutoff, btype='lowpass')

    filtered_signals = np.zeros_like(data)

    for n in range(data.shape[0]):
        for i in range(data.shape[2]):
            ecg_signal = data[n, :, i]
            coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
            cA = coeffs[0]
            filtered_cA = signal.filtfilt(b, a, cA)
            filtered_coeffs = [filtered_cA] + coeffs[1:]
            filtered_signal = pywt.waverec(filtered_coeffs, wavelet)
            filtered_signals[n, :, i] = filtered_signal

    return filtered_signals

# resampling ECG data
def resample_ecg_data(ecg_data, origianl_rate, target_rate, samples):
    """
    Resamples ECG data from 400 Hz to 500 Hz.

    Args:
        ecg_data (np.ndarray): ECG data with shape [N, 4096, 12].

    Returns:
        np.ndarray: Resampled ECG data with shape [N, M, 12], where M is the new number of samples after resampling.
    """
    # Compute the resampling ratio
    resampling_ratio = target_rate / origianl_rate

    # Compute the new number of samples after resampling
    M = int(ecg_data.shape[1] * resampling_ratio)

    # Initialize an array to store the resampled data
    ecg_data_resampled = np.zeros((ecg_data.shape[0], M, ecg_data.shape[2]))

    # Iterate over each channel and resample independently
    for i in range(ecg_data.shape[2]):
        for j in range(ecg_data.shape[0]):
            ecg_data_resampled[j, :, i] = resample(ecg_data[j, :, i], M)
    # Trim the resampled data to the last 4096 samples
    ecg_data_resampled = ecg_data_resampled[:, -samples:, :]
    return ecg_data_resampled

def set_channels_to_zero(ecg_data, n):
    """
    Randomly selects a number of ECG channels to set to zero for each group in the data.

    Args:
    - ecg_data: numpy array of shape (N, 4096, 12) containing ECG data
    - n: maximum number of channels that can be set to zero (up to n-1 channels can be left non-zero)

    Returns:
    - numpy array of shape (N, 4096, 12) with selected channels set to zero for each group
    """

    num_groups = 100
    # Choose number of channels to set to zero (up to n-1)
    num_channels_to_set_zero = n
    group_size = ecg_data.shape[0] // num_groups

    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size

        group_data = ecg_data[start_idx:end_idx, :, :]

        # Choose which channels to set to zero
        channels_to_set_zero = np.random.choice(group_data.shape[-1], num_channels_to_set_zero, replace=False)

        # Set selected channels to zero
        ecg_data[start_idx:end_idx, :, channels_to_set_zero] = 0

    # Handle the last group separately to avoid going beyond the shape
    start_idx = num_groups * group_size
    group_data = ecg_data[start_idx:, :, :]

    # Choose which channels to set to zero
    channels_to_set_zero = np.random.choice(group_data.shape[-1], num_channels_to_set_zero, replace=False)

    # Set selected channels to zero
    ecg_data[start_idx:, :, channels_to_set_zero] = 0

    return ecg_data


def min_max_normalize(x):
    # Get the shape of the input tensor
    batch_size, num_readings, num_channels = x.shape

    # Reshape the input tensor to (batch_size, num_readings * num_channels)
    x_flat = x.view(batch_size, -1)

    # Calculate the min and max values along the second dimension (num_channels)
    min_values = x_flat.min(dim=1, keepdim=True)[0]
    max_values = x_flat.max(dim=1, keepdim=True)[0]

    # Handle zero division by setting max_values and min_values to 1 for rows where all values are zero
    all_zeros = (min_values == 0) & (max_values == 0)
    max_values[all_zeros] = 1
    min_values[all_zeros] = 0

    # Normalize the data
    normalized_x_flat = (x_flat - min_values) / (max_values - min_values)

    # Reshape the normalized data back to the original shape
    normalized_x = normalized_x_flat.view(batch_size, num_readings, num_channels)

    return normalized_x


# Making prediction csv
# Open the HDF5 file
print('Reading Data')
path_to_hdf5 = '....hdf5'
hdf5_dset = 'tracings'
path_to_csv = '....csv'
f = h5py.File(path_to_hdf5, "r")
x = f[hdf5_dset][:]

# Read the CSV file
label = pd.read_csv(path_to_csv)[['1dAVb','RBBB','LBBB','AF']]
# Get the column names
columns = label.columns
# Convert label values to np.float32 data type
y = label.values.astype(np.float32)

label_model = ['1dAVb','RBBB','LBBB','SB','AF','ST']

print('Resampling X')
x = resample_ecg_data(x, 500, 400, 4096)
# print('Band passing X')
# x = apply_bandpass_filter(x)
# print('Filtering X')
# x = filter_ecg_signal(x)
print('Emptying X channels')
x = set_channels_to_zero(x, n)

print(x.shape, y.shape)


# Perform inference in batches
all_probs = []  # List to store probabilities for all batches

for start_idx in range(0, len(x), batch_size):
    end_idx = start_idx + batch_size
    batch_x_val = x[start_idx:end_idx]

    # Convert batch_x_val to a PyTorch tensor and move to the device
    batch_x_val_tensor = torch.tensor(batch_x_val, dtype=torch.float32, device=device)
    batch_x_val_tensor = min_max_normalize(batch_x_val_tensor)


    with torch.no_grad():
        outputs = nn.functional.sigmoid(model(batch_x_val_tensor))


    # Convert outputs to probabilities for the positive class
    all_probs.append(outputs.cpu().numpy())  # Store batch probabilities
    all_probs_array = np.concatenate(all_probs, axis=0)
    print(all_probs_array.shape)

# Initialize lists to store per-class metrics
class_metrics = []

for class_label in columns:  # Iterate through the column names
    if class_label in label_model:  # Only consider overlapping classes
        class_index = label_model.index(class_label)  # Get index for class in label_model
        class_index_y = columns.get_loc(class_label)

        y_val_binary = y[:, class_index_y]
        predictions = (all_probs_array[:, class_index] > 0.5).astype(int)

        accuracy = accuracy_score(y_val_binary, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val_binary, predictions, average='binary')
        auroc = roc_auc_score(y_val_binary, all_probs_array[:, class_index])

        class_metrics.append({
            'Class': class_label,
            'Recall': recall,
            'Precision': precision,
            'F1': f1,
            'AUROC': auroc,
        })

# Calculate average metrics
average_metrics = {
    'Class': 'Average',
    'Recall': np.mean([metric['Recall'] for metric in class_metrics]),
    'Precision': np.mean([metric['Precision'] for metric in class_metrics]),
    'F1': np.mean([metric['F1'] for metric in class_metrics]),
    'AUROC': np.mean([metric['AUROC'] for metric in class_metrics]),
}

# Append average metrics to the list of class metrics
class_metrics.append(average_metrics)

# Create a DataFrame to store the metrics
metrics_df = pd.DataFrame(class_metrics)

# Save the DataFrame to a CSV file
metrics_df.to_csv(output_directory + '/generalisation_' + str(n) + '.csv', index=False)

print('Completed')
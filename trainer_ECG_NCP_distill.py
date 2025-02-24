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

from sklearn.model_selection import train_test_split



# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Optimizer
parser.add_argument('--file_name', default='test', type=str, help='Folder Name')
parser.add_argument('--lr', default=0.003, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
# Scheduler
parser.add_argument('--epochs1', default=50, type=int, help='Training epochs')
parser.add_argument('--epochs2', default=300, type=int, help='Full Training epochs')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
# Model
parser.add_argument('--n_layers', default=1, type=int, help='Number of layers')
parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.1
                    , type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')

parser.add_argument('--cuda', default=0, type=int, help='Cuda')

parser.add_argument('--alpha', default=0.8, type=float, help='Alpha, learning rate of Distill')


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)


# Define the directory path and file name where you want to save the text file
output_directory = '...' + args.file_name
output_filename = 'argparse_config.txt'

if not os.path.exists(output_directory):
    # If it doesn't exist, create the directory
    os.makedirs(output_directory)
    print(f"Directory '{output_directory}' created successfully.")
else:
    print(f"Directory '{output_directory}' already exists.")


output_filepath = f'{output_directory}/{output_filename}'

# Write the parsed arguments to a text file
with open(output_filepath, 'w') as file:
    for arg, value in vars(args).items():
        file.write(f'{arg}: {value}\n')

print(f'Arguments saved to {output_filepath}')



n_layers = args.n_layers
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print(f'==> Preparing data..')

# with h5py.File('/mnt/data13_16T/jim/ECG_data/CPSC/CPSC.hdf5', 'r') as f:
with h5py.File('....hdf5', 'r') as f:
    X = f['tracings'][:, :, :].reshape(-1, 4096, 12) # [:]  # shape (B, 4096, 12)
    y = pd.read_csv('....csv').values.reshape(-1, 6)  # shape (B, 8)

    print(X.shape, y.shape)

# Define a custom PyTorch dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, seed=42):
        self.X = X
        self.y = y
        self.seed = seed
        np.random.seed(self.seed)
        self.indices = np.random.permutation(len(self.X))

    def __getitem__(self, index):
        # Get the input feature and target label for the given index
        idx = self.indices[index]
        x = self.X[idx].astype(np.float32)
        label = self.y[idx].astype(np.float32)
        return torch.tensor(x), torch.tensor(label)


    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.X)





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


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the train, validation, and test datasets
trainset = MyDataset(X_train, y_train)
valset = MyDataset(X_val, y_val)
testset = MyDataset(X_val, y_val)

d_input = 12
d_output = 6

# Dataloaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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
        # Decode the outputs
        x , _ = self.decoder(x)  # (B, d_model) -> (B, d_output)
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

model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True



if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

def setup_optimizer(model, lr, weight_decay, epochs):

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

class_weights = torch.tensor([6], dtype=torch.float32, device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
criterion2 = nn.BCEWithLogitsLoss()

optimizer, scheduler = setup_optimizer(
    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs2
)



def is_non_trainable_layer(name):
    # Layers to freeze (non-trainable)
    return (
        "encoder" in name or
        "decoder" in name or
        name in {
            "decoder.rnn_cell.layer_0.sparsity_mask",
            "decoder.rnn_cell.layer_1.sparsity_mask",
            "decoder.rnn_cell.layer_2.sparsity_mask"
        }
    )



###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################


precision_list_train_epoch = []
recall_list_train_epoch = []
specificity_list_train_epoch = []
accuracy_list_train_epoch = []
loss_train_epoch =[]
auroc_list_train_epoch = []

# Training
def train(use_alt_criterion=False):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    epsilon = 1e-8

    precision_list_train = []
    recall_list_train = []
    specificity_list_train = []
    accuracy_list_train = []
    auroc_list_train = []  # New list for AUROC

    targets_flat_all = []  # New list to record targets for the entire epoch
    outputs_flat_all = []  # New list to record outputs for the entire epoch



    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = min_max_normalize(inputs)
        optimizer.zero_grad()


            # **Use alternative criterion if the flag is True**
        if use_alt_criterion:
            outputs = model(inputs)
            layer_weights_before = {}

            # Clone the layers
            for name, param in model.named_parameters():
                if is_non_trainable_layer(name):
                    layer_weights_before[name] = param.clone().detach()

            # Normal Training
            loss = criterion(outputs, targets)  # Alternative loss function
            loss.backward()
            optimizer.step()

            # Getting model output
            output_temp = nn.functional.sigmoid(model(inputs)).detach()

            # **Restore non-trainable layer weights**
            with torch.no_grad():  # Ensures no gradient tracking
                for name, param in model.named_parameters():
                    if name in layer_weights_before:
                        param.copy_(layer_weights_before[name]) #.to(param.device))  # Copy back to the model

            # Freeze the layers
            for name, param in model.named_parameters():
                is_frozen = is_non_trainable_layer(name)
                param.requires_grad = not is_frozen

            # Self-Distilled Training
            optimizer.zero_grad()
            outputs = model(inputs)
            alpha = args.alpha
            loss = (1-alpha) * criterion(outputs, targets) + alpha * criterion2(outputs, output_temp) # Alternative loss function
            loss.backward()
            optimizer.step()

            # Unfreeze the layers
            for name, param in model.named_parameters():
                # Only allow the top and bottom layers to train in the next first part of the epoch
                is_frozen = is_non_trainable_layer(name)
                param.requires_grad = is_frozen



        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


        outputs = nn.functional.sigmoid(outputs)

        train_loss += loss.item()
        predicted = outputs.gt(0.5).long()
        tp = torch.zeros(d_output, dtype=torch.float)
        fp = torch.zeros(d_output, dtype=torch.float)
        fn = torch.zeros(d_output, dtype=torch.float)
        tn = torch.zeros(d_output, dtype=torch.float)
        for c in range(d_output):
            tp[c] = ((predicted[:, c] == 1) & (targets[:, c] == 1)).sum().item()
            fp[c] = ((predicted[:, c] == 1) & (targets[:, c] == 0)).sum().item()
            fn[c] = ((predicted[:, c] == 0) & (targets[:, c] == 1)).sum().item()
            tn[c] = ((predicted[:, c] == 0) & (targets[:, c] == 0)).sum().item()
        precision = tp.sum() / (tp.sum() + fp.sum()+ epsilon)
        recall = tp.sum() / (tp.sum() + fn.sum()+ epsilon)
        specificity = tn.sum() / (tn.sum() + fp.sum()+ epsilon)
        total += targets.size(0) * d_output
        correct += predicted.eq(targets).sum().item()

        # Flatten targets and outputs for the batch
        targets_flat = targets.cpu().detach().numpy().flatten()
        outputs_flat = outputs.cpu().detach().numpy().flatten()

        # Record targets and outputs for the batch
        targets_flat_all.extend(targets_flat.tolist())
        outputs_flat_all.extend(outputs_flat.tolist())

        # Calculate AUROC for the batch if there are more than one class
        if len(np.unique(targets_flat_all)) > 1:
            auroc_batch = roc_auc_score(targets_flat_all, outputs_flat_all)
        else:
            auroc_batch = None

        # Append AUROC for the batch to the list if it is calculated
        if auroc_batch is not None:
            auroc_list_train.append(auroc_batch)

        precision_list_train.append(precision.item())
        recall_list_train.append(recall.item())
        specificity_list_train.append(specificity.item())
        accuracy_list_train.append(correct / total)

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d) | Precision: %.3f | Recall: %.3f | Specificity: %.3f | AUROC: %s' %
            (batch_idx+1, len(trainloader), train_loss / (batch_idx + 1), 100. * correct / total, correct, total,
             precision, recall, specificity, str(round(auroc_batch, 3)) if auroc_batch is not None else "N/A")
        )

    # calculate metrics for the entire epoch
    precision_epoch = sum(precision_list_train) / len(precision_list_train)
    recall_epoch = sum(recall_list_train) / len(recall_list_train)
    specificity_epoch = sum(specificity_list_train) / len(specificity_list_train)
    accuracy_epoch = sum(accuracy_list_train) / len(accuracy_list_train)
    average_train_loss = train_loss / (batch_idx + 1)

    # Calculate mean AUROC for each class across all epochs
    auroc_epoch = auroc_list_train[-1]

    # record the metrics for the epoch
    precision_list_train_epoch.append(precision_epoch)
    recall_list_train_epoch.append(recall_epoch)
    specificity_list_train_epoch.append(specificity_epoch)
    accuracy_list_train_epoch.append(accuracy_epoch)
    auroc_list_train_epoch.append(auroc_epoch)
    loss_train_epoch.append(average_train_loss)


precision_list_eval_epoch = []
recall_list_eval_epoch = []
specificity_list_eval_epoch = []
accuracy_list_eval_epoch = []
loss_eval_epoch =[]
auroc_list_eval_epoch = []


def eval(epoch, dataloader, checkpoint=False):
    global best_acc
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    epsilon = 1e-8

    precision_list_eval = []
    recall_list_eval = []
    specificity_list_eval = []
    accuracy_list_eval = []
    auroc_list_eval = []

    targets_flat_all = []  # New list to record targets for the entire epoch
    outputs_flat_all = []  # New list to record outputs for the entire epoch

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = min_max_normalize(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            outputs = nn.functional.sigmoid(outputs)

            eval_loss += loss.item()
            predicted = outputs.gt(0.5).long()
            tp = torch.zeros(d_output, dtype=torch.float).to(device)
            fp = torch.zeros(d_output, dtype=torch.float).to(device)
            fn = torch.zeros(d_output, dtype=torch.float).to(device)
            tn = torch.zeros(d_output, dtype=torch.float).to(device)
            for c in range(d_output):
                tp[c] = ((predicted[:, c] == 1) & (targets[:, c] == 1)).sum().item()
                fp[c] = ((predicted[:, c] == 1) & (targets[:, c] == 0)).sum().item()
                fn[c] = ((predicted[:, c] == 0) & (targets[:, c] == 1)).sum().item()
                tn[c] = ((predicted[:, c] == 0) & (targets[:, c] == 0)).sum().item()
            precision = tp.sum() / (tp.sum() + fp.sum()+ epsilon)
            recall = tp.sum() / (tp.sum() + fn.sum()+ epsilon)
            specificity = tn.sum() / (tn.sum() + fp.sum()+ epsilon)
            total += targets.size(0) * d_output
            correct += predicted.eq(targets).sum().item()
            # Flatten targets and outputs for the batch
            targets_flat = targets.cpu().detach().numpy().flatten()
            outputs_flat = outputs.cpu().detach().numpy().flatten()

            # Record targets and outputs for the batch
            targets_flat_all.extend(targets_flat.tolist())
            outputs_flat_all.extend(outputs_flat.tolist())

            if len(np.unique(targets_flat_all)) > 1:
                auroc_batch = roc_auc_score(targets_flat_all, outputs_flat_all)
            else:
                auroc_batch = None

            # Append AUROC for the batch to the list if it is calculated
            if auroc_batch is not None:
                auroc_list_eval.append(auroc_batch)

            precision_list_eval.append(precision.item())
            recall_list_eval.append(recall.item())
            specificity_list_eval.append(specificity.item())
            accuracy_list_eval.append(correct / total)
            # auroc_list_eval.append(auroc_batch)

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d) | Precision: %.3f | Recall: %.3f | Specificity: %.3f | AUROC: %s' %
                (batch_idx+1, len(dataloader), eval_loss / (batch_idx + 1), 100. * correct / total, correct, total,
                 precision, recall, specificity, str(round(auroc_batch, 3)) if auroc_batch is not None else "N/A")
            )

        # calculate metrics for the entire epoch
        precision_epoch = sum(precision_list_eval) / len(precision_list_eval)
        recall_epoch = sum(recall_list_eval) / len(recall_list_eval)
        specificity_epoch = sum(specificity_list_eval) / len(specificity_list_eval)
        accuracy_epoch = sum(accuracy_list_eval) / len(accuracy_list_eval)
        average_eval_loss = eval_loss / (batch_idx + 1)

        # Append AUROC for the batch to the list
        auroc_list_eval.append(auroc_batch)

        # Calculate mean AUROC for each class across all epochs
        auroc_epoch = auroc_list_eval[-1]

        # record the metrics for the epoch
        precision_list_eval_epoch.append(precision_epoch)
        recall_list_eval_epoch.append(recall_epoch)
        specificity_list_eval_epoch.append(specificity_epoch)
        accuracy_list_eval_epoch.append(accuracy_epoch)
        auroc_list_eval_epoch.append(auroc_epoch)
        loss_eval_epoch.append(average_eval_loss)


    # Save checkpoint.
    if checkpoint:
        acc = 100.*correct/total
        if acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            # Check if the directory exists
            directory_path = './checkpoint/' + args.file_name
            if not os.path.exists(directory_path):
                # If it doesn't exist, create the directory
                os.makedirs(directory_path)
                print(f"Directory '{directory_path}' created successfully.")
            else:
                print(f"Directory '{directory_path}' already exists.")

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, directory_path + '/ckpt_' + str(args.n_layers) + '.pth')
            best_acc = acc

        return acc


# specify the csv file path
directory_path = output_directory
if not os.path.exists(directory_path):
    # If it doesn't exist, create the directory
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created successfully.")
else:
    print(f"Directory '{directory_path}' already exists.")



# Load the saved model from .pt file
state_dict = torch.load('....pt')

# Load the state dictionary into the model
model.load_state_dict(state_dict)
model = model.to(device)


for _ in range(args.epochs1):
    optimizer.step()  # First step optimizer (optional if no updates needed)
    scheduler.step()  # Move scheduler forward


pbar = tqdm(range(args.epochs1, args.epochs2))
val_acc = 0.0
for epoch in pbar:
    if epoch == 0:
        pbar.set_description('Epoch: %d' % (epoch))
    else:
        pbar.set_description('Epoch: %d | Val acc: %1.3f' % (epoch, val_acc))
    train(use_alt_criterion=True)
    val_acc = eval(epoch, valloader, checkpoint=True)
    scheduler.step()
    print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")


torch.save(model.state_dict(), '....pt')


# create a list of dictionaries containing the evaluation metrics
eval_metrics = [{'loss': l, 'precision': p, 'recall': r, 'specificity': s, 'accuracy': a, 'AUROC': u}
                for l, p, r, s, a, u in zip(loss_eval_epoch, precision_list_eval_epoch, recall_list_eval_epoch, specificity_list_eval_epoch, accuracy_list_eval_epoch, auroc_list_eval_epoch)]


csv_file = directory_path + '/brazil_evaluation_metrics_' + str(n_layers) + '_' + str(args.lr) +'.csv'

# write the evaluation metrics to the csv file
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['loss', 'precision', 'recall', 'specificity', 'accuracy', 'AUROC'])
    writer.writeheader()
    writer.writerows(eval_metrics)



# create a list of dictionaries containing the training metrics
train_metrics = [{'loss': l, 'precision': p, 'recall': r, 'specificity': s, 'accuracy': a, 'AUROC': u}
                 for l, p, r, s, a, u in zip(loss_train_epoch, precision_list_train_epoch, recall_list_train_epoch, specificity_list_train_epoch, accuracy_list_train_epoch, auroc_list_train_epoch)]

# specify the csv file path
csv_file = directory_path + '/brazil_training_metrics_' + str(n_layers) + '_' + str(args.lr) +'.csv'

# write the training metrics to the csv file
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['loss', 'precision', 'recall', 'specificity', 'accuracy', 'AUROC'])
    writer.writeheader()
    writer.writerows(train_metrics)

print('Completed')
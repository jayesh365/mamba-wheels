
import torch
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm.auto import tqdm

import csv

import numpy as np
import matplotlib.pyplot as plt
import wandb


class AlternatingSignalDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def generate_trace_task(num_seq=1000, n=200, two_peaks=False, holdout_intervals=None):
    holdout_inputs = []
    holdout_targets = []
    inputs = []
    targets = []
    # two_peaks = False
    # np.random.seed(102002)

    for seq_num in range(num_seq):
        signal = []
        length = 0
        
        # Set on/off lengths and peak distances based on `n`
        if n == 200:
            if not two_peaks:
                off_length = np.random.randint(20, 40)
                on_length = 20
            else:
                # off_length = np.random.randint(20, 70)
                off_length = 20
                peak_distance = 50
                on_length = 10

        elif n == 600:
            if not two_peaks:
                off_length = np.random.randint(60, 120)
                on_length = 60
                
            else:
                off_length = 200
                peak_distance = 200
                on_length = 30
            
        elif n == 1000:
            if not two_peaks:
                off_length = np.random.randint(100, 200)
                on_length = 100
                
            else:
                off_length = 400
                peak_distance = 400
                
                on_length = 50
        else:
            raise ValueError("Unsupported value of n. Choose 200, 600, or 1000.")
        
        # Generate the signal with one or two peaks based on `two_peaks`
        signal.extend([0] * off_length)
        signal.extend([1] * on_length)

        if two_peaks:
            # Add the specified distance between peaks, followed by the second peak
            signal.extend([0] * peak_distance)
            signal.extend([1] * on_length)

        # Fill with 0s until the signal length matches `n`
        signal.extend([0] * (n - len(signal)))
        
        # Trim in case signal exceeds `n` (shouldn’t normally happen with the fill above)
        # print('length of signal: ', len(signal))
        signal = signal[:n]

        # Convert signal to tensor format
        ts = torch.tensor(signal, dtype=torch.float32).unsqueeze(-1)
        
        # Create input and target sequences
        input_seq = ts
        target_seq = torch.cat((torch.tensor([0]*1, dtype=torch.float32).unsqueeze(-1), ts[:-10]), dim=0)
        # print('input shape: ', input_seq.shape)

        # Check if the current sequence is part of the holdout intervals
        if holdout_intervals and seq_num in holdout_intervals:
            holdout_inputs.append(input_seq)
            holdout_targets.append(target_seq)
        else:
            inputs.append(input_seq)
            targets.append(target_seq)

    # print(f'\nNUMBER OF INPUTS {torch.stack(holdout_inputs).shape}\n')
    # print(f'\nShape of inputs: {torch.stack(inputs).shape}')
    return torch.stack(targets), torch.stack(inputs), torch.stack(holdout_targets), torch.stack(holdout_inputs)


def generate_inter_trial_interval(num_seq=1000, n=200, iti=(20, 40), isi=10, holdout_intervals=None):
    '''
    Generate data to train S4D model.
    Data is of the form: initially off for (20-40) time steps, 
    then on for 10 time steps, and repeats.
    The length of the signals should be 200 time steps.
    '''
    holdout_inputs = []
    holdout_targets = []
    inputs = []
    targets = []

    for seq_num in range(num_seq):
        signal = []
        length = 0
        
        while length < n:
            # off_length = np.random.randint(20, 40)
            off_length = np.random.randint(iti[0], iti[1])
            # off_length = np.random.randint(2, 8)

            # on_length = 10
            on_length = isi
            # on_length = 2
            signal.extend([0] * off_length)
            signal.extend([1] * on_length)
            length += off_length + on_length

        signal = signal[:n]
        ts = torch.tensor(signal, dtype=torch.float32).unsqueeze(-1)
        
        # Create input and target sequences
        input_seq = ts
        target_seq = torch.cat((ts[1:], torch.tensor([0], dtype=torch.float32).unsqueeze(-1)), dim=0)
        
        # Check if the current sequence is part of the holdout intervals
        if holdout_intervals and seq_num in holdout_intervals:
            holdout_inputs.append(input_seq)
            holdout_targets.append(target_seq)
        else:
            inputs.append(input_seq)
            targets.append(target_seq)

    return torch.stack(inputs), torch.stack(targets), torch.stack(holdout_inputs), torch.stack(holdout_targets)

def get_offset_indices(signal):
    """
    Find all indices where ISI turns off (transitions from 1 to 0) in a signal.
    
    Args:
        signal: A torch.Tensor of shape [200] or similar 1D tensor
    
    Returns:
        List of integers representing indices where the signal transitions from 1 to 0
    """
    # Ensure we're working with a 1D tensor with proper values
    if torch.is_tensor(signal):
        # If tensor has extra dimensions, flatten it
        if signal.dim() > 1:
            signal = signal.flatten()
        
        # Convert to numpy for easier processing
        signal_np = signal.detach().cpu().numpy()
    else:
        # If it's already a numpy array or list
        signal_np = np.array(signal)
    
    # Find transitions from 1 to 0
    # Using numpy operations for efficiency
    transitions = np.diff(signal_np.astype(int))
    # A transition from 1 to 0 will have a difference of -1
    offset_indices = np.where(transitions == -1)[0] + 1  # +1 because diff gives index before transition
    
    return offset_indices.tolist()


def get_offset_indices_batched(signals):
    """
    Find all indices where ISI turns off (transitions from 1 to 0) for a batch of signals.
    
    Args:
        signals: A torch.Tensor of shape [batch_size, sequence_length], e.g., [8, 200]
    
    Returns:
        List of lists, where each inner list contains the offset indices for one signal in the batch
    """
    if not torch.is_tensor(signals):
        signals = torch.tensor(signals)
    
    # Ensure the tensor is on CPU for numpy conversion
    if signals.device.type != 'cpu':
        signals = signals.detach().cpu()
    
    # Initialize list to store offsets for each signal in the batch
    batch_offsets = []
    
    # Process each signal in the batch
    for i in range(signals.shape[0]):
        # Extract single signal
        signal = signals[i]
        
        # Convert to numpy
        signal_np = signal.numpy()
        
        # Find transitions from 1 to 0 using numpy diff
        transitions = np.diff(signal_np.astype(int))
        
        # A transition from 1 to 0 will have a difference of -1
        offset_indices = np.where(transitions == -1)[0] + 1  # +1 because diff gives index before transition
        
        # Store the offsets for this signal
        batch_offsets.append(offset_indices.tolist())
    
    return batch_offsets


def get_flat_batch_indices(batch_tensor):
    """
    Get flat (batch_idx, seq_idx) pairs for all ISI offset points.
    
    Args:
        batch_tensor: A torch.Tensor of shape [batch_size, sequence_length]
    
    Returns:
        Two numpy arrays: 
        - batch_indices: array of batch indices
        - seq_indices: array of sequence indices where ISI turns off
    """
    # Get lists of offset indices for each sequence in the batch
    batch_offsets = get_offset_indices_batched(batch_tensor)
    
    # Prepare flat arrays
    batch_indices = []
    seq_indices = []
    
    # Collect all (batch_idx, seq_idx) pairs
    for batch_idx, offsets in enumerate(batch_offsets):
        for seq_idx in offsets:
            batch_indices.append(batch_idx)
            seq_indices.append(seq_idx)
    
    return np.array(batch_indices), np.array(seq_indices)


def check_batch_match_at_offset(binary_output, targets, match_threshold=0.5):
    """
    Checks if at least half of the batch from binary output matches the target at offset indices.
    
    Args:
        binary_output: Tensor of shape [batch_size, sequence_length] with binary values (0 or 1)
        targets: Tensor of shape [batch_size, sequence_length] or [batch_size, sequence_length, 1]
        match_threshold: Minimum fraction of matches required (default: 0.5 means at least 50%)
    
    Returns:
        tuple: (match_ratio, match_result, match_details)
            - match_ratio: Fraction of sequences with matching offsets
            - match_result: Boolean indicating if match_ratio >= match_threshold
            - match_details: Dictionary with detailed information about matches
    """
    # Get batched offset indices
    batch_offsets = get_offset_indices_batched(targets.squeeze())

    
    # Make sure targets has same shape as binary_output
    if targets.dim() > binary_output.dim():
        targets = targets.squeeze(-1)
    
    # Move tensors to CPU for processing
    if binary_output.device.type != 'cpu':
        binary_output = binary_output.detach().cpu()
    if targets.device.type != 'cpu':
        targets = targets.detach().cpu()
    
    # Initialize counters
    batch_size = binary_output.shape[0]
    sequences_with_matches = 0
    total_offsets = 0
    total_matches = 0
    match_details = {}
    
    # Check each sequence in the batch
    for batch_idx, offsets in enumerate(batch_offsets):
        if not offsets:  # Skip if no offsets found
            match_details[f"sequence_{batch_idx}"] = {
                "offsets": 0,
                "matches": 0,
                "match_ratio": 0,
                "status": "no_offsets"
            }
            continue
        
        # Count matches for this sequence
        sequence_matches = 0
        sequence_details = []
        
        for offset_idx in offsets:
            # The actual offset where ISI turns off
            offset = offset_idx
            
            # Check if value at offset matches in target
            # We expect 0 at offset since that's where ISI turns off
            # But we check preceding value to see if target also turns off at same point
            if offset > 0 and offset < binary_output.shape[1]:
                # Get the value before the transition (should be 1)
                prev_binary = binary_output[batch_idx, offset-1].item()
                prev_target = targets[batch_idx, offset-1].item()
                
                # Get the value at the transition (should be 0)
                curr_binary = binary_output[batch_idx, offset].item()
                curr_target = targets[batch_idx, offset].item()
                
                # Check if pattern matches: 1→0 transition in both binary_output and target
                transition_match = (prev_binary > 0.5 and prev_target > 0.5 and 
                                    curr_binary < 0.5 and curr_target < 0.5)
                
                if transition_match:
                    sequence_matches += 1
                
                sequence_details.append({
                    "offset": offset,
                    "binary_before": prev_binary,
                    "target_before": prev_target,
                    "binary_at": curr_binary, 
                    "target_at": curr_target,
                    "match": transition_match
                })
        
        # Calculate match ratio for this sequence
        seq_match_ratio = sequence_matches / len(offsets) if offsets else 0
        sequence_has_match = seq_match_ratio >= match_threshold
        
        if sequence_has_match:
            sequences_with_matches += 1
        
        # Store sequence details
        match_details[f"sequence_{batch_idx}"] = {
            "offsets": len(offsets),
            "matches": sequence_matches,
            "match_ratio": seq_match_ratio,
            "status": "match" if sequence_has_match else "no_match",
            "details": sequence_details
        }
        
        # Update totals
        total_offsets = len(offsets)
        total_matches += sequence_matches
    
    # Calculate overall match ratio
    batch_match_ratio = sequences_with_matches / batch_size if batch_size > 0 else 0
    overall_match_ratio = total_matches / total_offsets if total_offsets > 0 else 0
    
    # Add summary to match details
    match_details["summary"] = {
        "batch_size": batch_size,
        "sequences_with_matches": sequences_with_matches,
        "batch_match_ratio": batch_match_ratio,
        "total_offsets": total_offsets,
        "total_matches": total_matches,
        "overall_match_ratio": overall_match_ratio,
        "threshold_met": batch_match_ratio >= match_threshold
    }
    
    return batch_match_ratio, batch_match_ratio >= match_threshold, match_details


# Split train data into train and validation sets
def split_train_val_og(train_inputs, train_targets, val_split=0.1):
    dataset_size = len(train_inputs)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_inputs_split = train_inputs[train_indices]
    train_targets_split = train_targets[train_indices]
    
    val_inputs_split = train_inputs[val_indices]
    val_targets_split = train_targets[val_indices]

    return train_inputs_split, train_targets_split, val_inputs_split, val_targets_split



# split dataset into train and validation sets
def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val



def visualize_signals(input_signal, target_signal):
    
    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True, figsize=(8, 8))

    # ax = axes[0]
    # ax.plot(input_signal.numpy(), 'r', label=' ITI ')
    # ax.plot(target_signal.numpy(), 'b', label=' ISI ')

    ax.plot(input_signal.numpy(), 'r', label='Input')
    ax.plot(target_signal.numpy(), 'b', label='Target')

    # labelLines(ax.get_lines(), zorder=5.5, color='r')
    

    plt.legend()    
    plt.title('Input Signal')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.tight_layout()
    # plt.show()
    plt.savefig('./test_out.png')

# train model
def train(model, trainloader, device, optimizer, criterion, epoch, model_name, wand, clip_grad):
    model.train()
    train_loss = 0
    total_mse = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))

    for batch_i, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        # print('\n', '='*20)
        # print('\ninputs shape: ', inputs.shape)
        # print('\ntargets shape: ',targets.shape)
        # print('='*20, '\n')
        optimizer.zero_grad()

        # if model_name == 'S4D' or model_name == 'IDS4': outputs = model(inputs).unsqueeze(-1)
        # else: outputs = model(inputs)
        
        if model_name == 'Transformer':
            src_mask =  nn.Transformer.generate_square_subsequent_mask(inputs.size(1)).to(device)
            outputs = model(inputs, src_mask=src_mask)
            # visualize_signals(inputs[1].detach().cpu(), torch.sigmoid(outputs[1].detach().cpu()))
            
        else: 
            outputs = model(inputs)
            # print(outputs[1])
            # visualize_signals(inputs[1].detach().cpu(), torch.sigmoid(outputs[1].detach().cpu()))
            
        # print('\n', '='*20)
        # print('\ninputs shape: ', inputs.shape)
        # print('\ntargets shape: ',targets.shape)
        # print('\noutputs shape: ',outputs.shape)
        # print('='*20, '\n')
        loss = criterion(outputs.squeeze(), targets.squeeze())
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        train_loss += loss.item()

        # put sigmoid for mse on outputs

        mse = F.mse_loss(torch.sigmoid(outputs), targets).item()
        # total_mse += mse

        total += targets.size(0)

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | MSE: %.3f' %
            (batch_i, len(trainloader), train_loss/(batch_i+1), mse)
        )
        if wand:
            wandb.log({"train_loss": train_loss / (batch_i + 1), "train_mse": mse,"epoch": epoch})


def eval(model, valloader, device, criterion, epoch, model_name, wand, clip_grad):

    model.eval()
    eval_loss = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(valloader))

        for batch_ind, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            # print('valloader ', inputs.shape, targets.shape)
            
            # if model_name == 'S4D' or model_name == 'IDS4': outputs = model(inputs).unsqueeze(-1)
            # else: outputs = model(inputs)
            if model_name == 'Transformer':
                src_mask =  nn.Transformer.generate_square_subsequent_mask(inputs.size(1)).to(device)
                outputs = model(inputs, src_mask=src_mask)
            else: 
                # print('hkadkhjsdkh', inputs.shape)
                
                outputs = model(inputs)
                # print('\n output shape: ', outputs.shape)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            mse = F.mse_loss(torch.sigmoid(outputs), targets).item()

            eval_loss += loss.item()
            total += inputs.size(0)
            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f' %
                (batch_ind, len(valloader), eval_loss/(batch_ind+1))
            )

            if wand:
                wandb.log({"val_loss": eval_loss / (batch_ind + 1), "val_mse": mse, "epoch": epoch})


def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # print('all parameters: ', all_parameters)

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
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
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



def see_all():
    '''
    viz all train and test data in one image
    '''

    test_ckpt = torch.load('./datasets/trace/test.pth')


    # print(test_ckpt.dataset.__len__())
    test_inputs = test_ckpt.dataset.inputs
    test_targets = test_ckpt.dataset.targets
    # Number of subplots (assuming you want one plot for each input-target pair)
    num_plots = len(test_inputs) + len(test_targets)
    
    # Determine grid size (e.g., square or close to square)
    grid_size = int(np.ceil(np.sqrt(num_plots)))

    # Create subplots grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))  # Adjust the figsize as needed
    axes = axes.flatten()  # Flatten in case grid_size doesn't match perfectly

    # Plot each input in a separate subplot
    for idx, inputs in enumerate(test_inputs):
        axes[idx].plot(inputs)
        axes[idx].set_title(f'Input {idx+1}')

    # Plot each target in a separate subplot, continuing from where inputs left off
    for idx, targets in enumerate(test_targets):
        axes[len(test_inputs) + idx].plot(targets)
        axes[len(test_inputs) + idx].set_title(f'Target {idx+1}')

    # Hide any remaining empty subplots if number of plots isn't a perfect square
    for ax in axes[num_plots:]:
        ax.axis('off')

    plt.tight_layout()  # Ensure plots are neatly organized
    plt.show()


def model_params(seed, model_checkpoint, model_type, hidden, output_file):
    
    num_layers = 1
    input_size = 1
    hidden_size = int(hidden)
    
    # Print the model details (optional)
    print(model_type, input_size, hidden_size, num_layers)
    
    # Initialize the model (assuming get_model is defined elsewhere)
    model = get_model(model_type, input_size, hidden_size, num_layers)
    
    # Load model from checkpoint
    checkpoint = torch.load(model_checkpoint)
    
    # Check if the checkpoint contains 'model_state_dict'
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")
    else:
        print("Checkpoint does not contain 'model_state_dict'.")
        total_params = 'N/A'  # Default value if state_dict is not found
    
    # Prepare data to write into the CSV file
    data = [model_type, input_size, hidden_size, num_layers, total_params, seed]

    # Write the parameters into the CSV file
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Check if file is empty to write the header
        file.seek(0, 2)  # Move the pointer to the end of the file
        if file.tell() == 0:
            writer.writerow(['Model Type', 'Input Size', 'Hidden Size', 'Num Layers', 'Total Params', 'Seed'])
        
        # Write the model details
        writer.writerow(data)

    print(f"Model parameters saved to {output_file}.")

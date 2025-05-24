"""
Model Evaluation Module
Easy-to-import module for running model evaluations in Google Colab
"""

from pathlib import Path
import re
import torch
from torch import nn
import numpy as np
import os
from typing import Dict, Tuple, Optional, Union, List
import ast

# Import model classes
try:
    from mamba_bits import Mamba_model as Mamba
    from s4d_bits import S4DTokenClassifier as S4D
    print("Successfully imported Mamba and S4D models")
    print(f"Mamba: {Mamba}")
    print(f"S4D: {S4D}")
except ImportError as e:
    print(f"Failed to import model classes: {e}")
    print("Make sure mamba_bits and s4d_bits are available")
    Mamba = None
    S4D = None

try:
    from helpers.auxs import (
        generate_trace_task, split_train_val_og,
        generate_inter_trial_interval, 
        get_flat_batch_indices, check_batch_match_at_offset
    )
except ImportError as e:
    print(f"Failed to import helpers: {e}")
    print("Make sure the helpers directory is in your PYTHONPATH")

def get_model(model_name, input_size, hidden_size, state_size, num_layers):
    """Initialize and return a model based on the model name."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_name == 'S4D':
        if S4D is None:
            raise ImportError("S4D model class not available")
        model = S4D(
            d_model=hidden_size,
            n_layers=num_layers,
            n_vocab=input_size,
            dropout=0,
            d_state=state_size,
            embed=True
        )
    elif model_name == 'MAMBA':
        if Mamba is None:
            raise ImportError("Mamba model class not available")
        model = Mamba(
            d_model=hidden_size,
            d_state=state_size,
            n_layers=1,
            n_vocab=input_size,
            dropout=0,
            embed=True
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model = model.to(device)
    return model

def load_test_data(ts_length, data_path):
    """Load test dataset for evaluation"""
    print(f"Loading test data from: {data_path}")
    testset_path = Path(data_path)
    print(f"Loading test data from: {testset_path}")

    if not testset_path.exists():
        raise FileNotFoundError(f"Test data file not found: {testset_path}")

    test_data = torch.load(testset_path, weights_only=False)
    print(f"Test data loaded successfully")
    return test_data

def setup_environment() -> str:
    """Set up environment and return device."""
    import warnings
    warnings.filterwarnings('ignore')
    
    # Set random seed for reproducibility
    np.random.seed(192390)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    return device

def extract_model_info(file_path: str) -> Optional[Dict[str, Union[str, int, float, Path]]]:
    """Extract model information from checkpoint file path."""
    # Updated pattern to handle scientific notation in learning rates (e.g., 1e-05)
    pattern = r'(\w+)_([\d\.e\-]+)_([\d]+)_64_([\d]+)_ckpt_([\d]+)_([\d]+)\.pth'
    
    # Convert to Path object if it's a string
    path_obj = Path(file_path)
    
    # Check if file exists
    if not path_obj.exists():
        print(f"Checkpoint file does not exist: {file_path}")
        return None
    
    match = re.search(pattern, path_obj.name)
    
    if match:
        model_info = {
            'model': match.group(1),
            'learning_rate': float(match.group(2)),
            'hidden_size': int(match.group(3)),
            'epochs': int(match.group(4)),
            'checkpoint_id': int(match.group(5)),
            'date': match.group(6),
            'matching_file': file_path
        }
        print(f"Extracted model info: {model_info['model']}, lr={model_info['learning_rate']}, "
              f"hidden_size={model_info['hidden_size']}, checkpoint_id={model_info['checkpoint_id']}, "
              f"date={model_info['date']}")
        return model_info
    else:
        # Try the old pattern as a fallback
        old_pattern = r'(\w+)_([\d]+)_64_([\d]+)_ckpt_([\d]+)'
        match = re.search(old_pattern, path_obj.name)
        
        if match:
            model_info = {
                'model': match.group(1),
                'hidden_size': int(match.group(2)),
                'epochs': int(match.group(3)),
                'checkpoint_id': int(match.group(4)),
                'matching_file': file_path
            }
            print(f"Extracted model info (old format): {model_info['model']}, "
                  f"hidden_size={model_info['hidden_size']}, checkpoint_id={model_info['checkpoint_id']}")
            return model_info
        else:
            print(f"Could not extract model info from {file_path} - file name doesn't match expected pattern")
            return None

def generate_data(seq_len: int, phase: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate data based on sequence length and phase alignment."""
    print(f"Generating data with seq_len={seq_len}, phase={phase}")
    
    # Generate random holdout intervals
    np.random.seed(192390)
    holdout_intervals = np.random.randint(0, 99, size=10).tolist()
    phase_tuple = ast.literal_eval(phase)
    
    try:
        if seq_len == 200:
            if phase_tuple[0]:
                train_inputs, train_targets, test_inputs, test_targets = generate_trace_task(1000, 200, phase_tuple[1], holdout_intervals=holdout_intervals)
            else:
                train_inputs, train_targets, test_inputs, test_targets = generate_inter_trial_interval(1000, 200, holdout_intervals=holdout_intervals)
        elif seq_len == 600:
            if phase_tuple[0]:
                train_inputs, train_targets, test_inputs, test_targets = generate_trace_task(1000, 600, phase_tuple[1], holdout_intervals=holdout_intervals)
            else:
                train_inputs, train_targets, test_inputs, test_targets = generate_inter_trial_interval(num_seq=1000, n=600, iti=(60, 120), isi=30, holdout_intervals=holdout_intervals)
        elif seq_len == 1000:
            if phase_tuple[0]:
                train_inputs, train_targets, test_inputs, test_targets = generate_trace_task(1000, 1000, phase_tuple[1], holdout_intervals=holdout_intervals)
            else:
                train_inputs, train_targets, test_inputs, test_targets = generate_inter_trial_interval(num_seq=1000, n=1000, iti=(100, 120), isi=50, holdout_intervals=holdout_intervals)
        else:
            raise ValueError(f"Unsupported sequence length: {seq_len}")
        
        print(f'train inputs {train_inputs.shape} {train_targets.shape}')
        print(f'test inputs {test_inputs.shape} {test_targets.shape}')
        return train_inputs, train_targets, test_inputs, test_targets
    except Exception as e:
        print(f"Error generating data: {e}")
        raise

def evaluate_model(
    model_info: Dict[str, Union[str, int, float, Path]],
    test_inputs: torch.Tensor,
    device: str,
    current_seed: str
) -> Optional[torch.Tensor]:
    """Evaluate a model and return its outputs."""
    model_name = model_info['model']
    file_path = model_info['matching_file']
    hidden_size = model_info['hidden_size']
    
    print(f"Evaluating model {model_name} (hidden_size={hidden_size}) from {file_path}")
    
    try:
        # Try to load the checkpoint
        try:
            checkpoint = torch.load(file_path, map_location=device)
            print(f"Successfully loaded checkpoint")
        except Exception as e:
            print(f"Failed to load checkpoint {file_path}: {e}")
            return None
        
        # Try to initialize the model
        try:
            print(f'model {model_name} hidden size {hidden_size}')
            model = get_model(model_name, 1, hidden_size, 64, 1).to(device)
            
            # print model parameters
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f'Layer: {name}, Size: {param.size()}')
            print(f"Successfully initialized model")
        except Exception as e:
            print(f"Failed to initialize model {model_name}: {e}")
            return None
        
        # Try to load the state dict
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded state dict")
        except KeyError:
            print("Checkpoint does not contain 'model_state_dict' key")
            if 'state_dict' in checkpoint:
                print("Trying 'state_dict' key instead")
                try:
                    model.load_state_dict(checkpoint['state_dict'])
                    print("Successfully loaded state dict using 'state_dict' key")
                except Exception as e:
                    print(f"Failed to load alternative state dict: {e}")
                    return None
            else:
                print("No usable state dict found in checkpoint")
                return None
        except Exception as e:
            print(f"Failed to load state dict: {e}")
            return None
        
        # Evaluate the model
        model.eval()
        with torch.no_grad():
            try:
                if model_name == 'Transformer':
                    src_mask = nn.Transformer.generate_square_subsequent_mask(test_inputs.size(1)).to(device)
                    outputs = torch.sigmoid(model(test_inputs, src_mask=src_mask)).cpu().detach().squeeze()
                else:
                    outputs = torch.sigmoid(model(test_inputs)).cpu().detach().squeeze()
                
                print(f"Successfully evaluated model, output shape: {outputs.shape}")
                return outputs
            except Exception as e:
                print(f"Error during model inference: {e}")
                return None
    except Exception as e:
        print(f"Unexpected error evaluating model {model_name}: {e}")
        return None

def run_evaluation(
    checkpoint_paths: List[str] = None,
    seq_len: int = 200,
    phase: str = '(True, False)',
    sample_idx: int = 3
) -> Dict:
    """
    Main evaluation function that can be easily imported and run.
    
    Args:
        checkpoint_paths: List of checkpoint file paths. If None, uses defaults.
        seq_len: Sequence length for data generation
        phase: Phase configuration as string
        sample_idx: Sample index (currently unused but kept for compatibility)
    
    Returns:
        Dictionary containing all results and data for plotting
    """
    # Default checkpoint paths
    if checkpoint_paths is None:
        checkpoint_paths = [
            '/content/mamba-wheels/model_ckpts/MAMBA_0.01_8_64_1000_ckpt_3321_20250423.pth',
            '/content/mamba-wheels/model_ckpts/S4D_0.005_8_64_1000_ckpt_879965_20250408.pth'
        ]
    
    # Determine phase type for logging
    if phase == str((True, False)):
        type_phase = 'double'
    elif phase == str((True, True)):
        type_phase = 'conset'
    elif phase == str((False, False)):
        type_phase = 'offset'
    else:
        type_phase = 'None'
    
    # Log the configuration
    print(f"Configuration: seq_len={seq_len}, phase={phase}, sample_idx={sample_idx}")
    print(f"Number of checkpoint paths: {len(checkpoint_paths)}")
    
    # Setup
    device = setup_environment()
    
    # Generate data
    train_inputs, train_targets, test_inputs, test_targets = generate_data(seq_len, phase)
    
    # Split data
    try:
        train_inputs_split, train_targets_split, val_inputs_split, val_targets_split = split_train_val_og(
            train_inputs, train_targets, val_split=0.1
        )
    except Exception as e:
        print(f"Error splitting data: {e}")
        return None
    
    # Move test data to device
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)
    
    # Create dictionaries to store outputs and model-to-seed mapping
    all_outputs_dict = {}
    model_to_seed_map = {}
    
    # Process each checkpoint file
    for checkpoint_path in checkpoint_paths:
        print(f"Processing checkpoint: {checkpoint_path}")
        
        # Skip if file doesn't exist
        if not Path(checkpoint_path).exists():
            print(f"Checkpoint file does not exist: {checkpoint_path}")
            continue
        
        # Extract model info
        model_info = extract_model_info(checkpoint_path)
        if model_info is None:
            continue
        
        # Extract seed from the checkpoint path
        seed_match = re.search(r'_ckpt_(\d+)_', checkpoint_path)
        current_seed = seed_match.group(1) if seed_match else "unknown"
        
        # Evaluate model
        outputs = evaluate_model(model_info, test_inputs, device, current_seed)
        
        # Create a formatted key for the outputs dictionary that includes learning rate
        if 'learning_rate' in model_info:
            model_name = f"{model_info['model']}_{model_info['learning_rate']}_{model_info['checkpoint_id']}"
        else:
            model_name = f"{model_info['model']}_{model_info['checkpoint_id']}"
        
        if outputs is not None:
            all_outputs_dict[model_name] = outputs
            model_to_seed_map[model_name] = current_seed
    
    # Process results
    if all_outputs_dict:
        results_data = []
        
        for k, v in all_outputs_dict.items():
            binary_output = (v.to(device) >= 0.5).float()
            
            # Check if at least half of the batch matches at the offset positions
            match_ratio, threshold_met, match_details = check_batch_match_at_offset(
                binary_output,
                test_targets,
                match_threshold=0.5  # At least 50% need to match
            )
            
            # Get flat batch indices for reference
            batch_idx, seq_idx = get_flat_batch_indices(binary_output)
            
            # Print the high-level results
            print('='*50)
            print(f'Model: {k}')
            print(f"sequence length: {seq_len}")
            print(f"Match ratio: {match_ratio:.2f}")
            print(f"Threshold met: {threshold_met}")
            print(f"Sequences with matches: {match_details['summary']['sequences_with_matches']} out of {match_details['summary']['batch_size']}")
            print(f"Total matching offsets: {match_details['summary']['total_matches']} out of {match_details['summary']['total_offsets']}")
            print('\n')
            
            # Store data for plotting
            results_data.append({
                'model_name': k,
                'original_output': v.cpu().detach(),
                'binary_output': binary_output.cpu().detach(),
                'targets': test_targets.cpu().detach(),
                'match_ratio': match_ratio,
                'threshold_met': threshold_met,
                'match_details': match_details
            })
        
        return {
            'results_data': results_data,
            'all_outputs_dict': all_outputs_dict,
            'seq_len': seq_len,
            'type_phase': type_phase,
            'test_targets': test_targets.cpu().detach(),
            'model_to_seed_map': model_to_seed_map,
            'train_data': {
                'train_inputs': train_inputs,
                'train_targets': train_targets,
                'train_inputs_split': train_inputs_split,
                'train_targets_split': train_targets_split,
                'val_inputs_split': val_inputs_split,
                'val_targets_split': val_targets_split
            }
        }
    else:
        print("No models were successfully evaluated.")
        return None

# Convenience functions for common use cases
def evaluate_single_model(checkpoint_path: str, seq_len: int = 200, phase: str = '(True, False)'):
    """Evaluate a single model."""
    return run_evaluation([checkpoint_path], seq_len, phase)

def evaluate_default_models(seq_len: int = 200, phase: str = '(True, False)'):
    """Evaluate the default MAMBA and S4D models."""
    return run_evaluation(None, seq_len, phase)

def evaluate_custom_models(checkpoint_paths: List[str], seq_len: int = 200, phase: str = '(True, False)'):
    """Evaluate custom list of models."""
    return run_evaluation(checkpoint_paths, seq_len, phase)

# For backward compatibility
def main(checkpoint_path, seq_len, phase, sample_idx):
    """Legacy main function - redirects to run_evaluation."""
    return run_evaluation([checkpoint_path], seq_len, phase, sample_idx)
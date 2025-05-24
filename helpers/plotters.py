import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Union
import os

def plot_signals_with_offsets(
    original_output: torch.Tensor,
    binary_output: torch.Tensor,
    targets: torch.Tensor,
    model_name: str,
    num_samples: int = 4,
    ts_length: int = 200,
    type_phase: str = 'unknown'
) -> plt.Figure:
    """
    Plot original outputs, binary outputs, and targets with offset markers.
    
    Args:
        original_output: Raw model outputs (continuous values)
        binary_output: Thresholded binary outputs
        targets: Target values
        model_name: Name of the model for the title
        num_samples: Number of samples to plot
        ts_length: Length of time series
        type_phase: Type of phase for labeling
    
    Returns:
        matplotlib Figure object
    """
    # Ensure tensors are on CPU for plotting
    if isinstance(original_output, torch.Tensor):
        original_output = original_output.cpu().detach()
    if isinstance(binary_output, torch.Tensor):
        binary_output = binary_output.cpu().detach()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach()
    
    # Limit number of samples to what's available
    num_samples = min(num_samples, original_output.shape[0])
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    # Color scheme
    colors = {
        'original': 'blue',
        'binary': 'red', 
        'target': 'green',
        'offset_marker': 'orange'
    }
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Plot original output (continuous)
        ax.plot(original_output[i], 
               color=colors['original'], 
               alpha=0.7, 
               linewidth=2, 
               label='Original Output')
        
        # Plot binary output
        ax.plot(binary_output[i], 
               color=colors['binary'], 
               alpha=0.8, 
               linewidth=1.5, 
               label='Binary Output', 
               linestyle='--')
        
        # Plot targets
        ax.plot(targets[i], 
               color=colors['target'], 
               alpha=0.6, 
               linewidth=2, 
               label='Target', 
               linestyle=':')
        
        # Add offset markers (vertical lines where targets change from 0 to 1)
        try:
            target_diff = torch.diff(targets[i].float(), prepend=torch.tensor([0.0]))
            onset_indices = torch.where(target_diff > 0.5)[0]
            
            for onset_idx in onset_indices:
                ax.axvline(x=onset_idx.item(), 
                          color=colors['offset_marker'], 
                          linestyle=':', 
                          alpha=0.8, 
                          linewidth=2)
            
            if len(onset_indices) > 0 and i == 0:  # Add legend entry only once
                ax.axvline(x=-1, color=colors['offset_marker'], 
                          linestyle=':', label='Target Onsets', linewidth=2)
        except Exception as e:
            print(f"Warning: Could not add offset markers for sample {i}: {e}")
        
        # Formatting
        ax.set_xlim(0, ts_length)
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel('Signal Value')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Sample {i+1} - {model_name} ({type_phase})')
        
        # Add legend only to first subplot to avoid clutter
        if i == 0:
            ax.legend(loc='upper right', fontsize=10)
    
    # Set xlabel only for bottom subplot
    axes[-1].set_xlabel('Time Steps')
    
    # Overall title
    fig.suptitle(f'{model_name} - Signal Analysis ({type_phase})', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig

def setup_colors() -> dict:
    """Set up colors for different models."""
    return {
        'LSTM': plt.get_cmap("tab10")(0),  # Blue
        'GRU': plt.get_cmap("tab10")(1),   # Orange
        'RNN': plt.get_cmap("tab10")(2),   # Green
        'Transformer': plt.get_cmap("tab10")(3),  # Red
        'S4D': plt.get_cmap("tab10")(4),   # Purple
        'MAMBA': plt.get_cmap("tab10")(5),  # Brown
        'NRU': plt.get_cmap("tab10")(6),   # Pink
    }

def plot_comparison_summary(
    all_outputs_dict: dict,
    targets: torch.Tensor,
    seq_len: int,
    type_phase: str,
    sample_idx: int = 0
) -> plt.Figure:
    """
    Create a summary comparison plot of all models for a single sample.
    
    Args:
        all_outputs_dict: Dictionary of model outputs
        targets: Target tensor
        seq_len: Sequence length
        type_phase: Phase type
        sample_idx: Which sample to plot
    
    Returns:
        matplotlib Figure object
    """
    colors = setup_colors()
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    # Plot targets first
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach()
    
    ax.plot(targets[sample_idx], 
           color='black', 
           linewidth=3, 
           label='Target', 
           alpha=0.8)
    
    # Plot each model's output
    for i, (model_name, outputs) in enumerate(all_outputs_dict.items()):
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().detach()
        
        # Get model type for coloring
        model_type = model_name.split('_')[0]
        color = colors.get(model_type, plt.get_cmap("tab10")(i % 10))
        
        # Plot binary output
        binary_output = (outputs >= 0.5).float()
        ax.plot(binary_output[sample_idx], 
               color=color, 
               linewidth=2, 
               label=f'{model_name} (Binary)',
               alpha=0.7,
               linestyle='--')
        
        # Plot original output with transparency
        ax.plot(outputs[sample_idx], 
               color=color, 
               linewidth=1, 
               alpha=0.4,
               linestyle='-')
    
    # Add onset markers
    try:
        target_diff = torch.diff(targets[sample_idx].float(), prepend=torch.tensor([0.0]))
        onset_indices = torch.where(target_diff > 0.5)[0]
        
        for onset_idx in onset_indices:
            ax.axvline(x=onset_idx.item(), 
                      color='red', 
                      linestyle=':', 
                      alpha=0.6, 
                      linewidth=1)
    except Exception as e:
        print(f"Warning: Could not add onset markers: {e}")
    
    ax.set_xlim(0, seq_len)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Signal Value')
    ax.set_title(f'Model Comparison - Sample {sample_idx + 1} ({type_phase})')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def plot_results(results_dict, save_plots=True, output_dir='./model_outputs/'):
    """
    Generate all plots from results dictionary returned by main().
    
    Args:
        results_dict: Dictionary returned by main() function
        save_plots: Whether to save plots to disk
        output_dir: Directory to save plots
    """
    results_data = results_dict['results_data']
    seq_len = results_dict['seq_len']
    type_phase = results_dict['type_phase']
    all_outputs_dict = results_dict['all_outputs_dict']
    test_targets = results_dict['test_targets']
    
    all_figures = []
    
    # Individual model plots
    for result in results_data:
        fig = plot_signals_with_offsets(
            original_output=result['original_output'],
            binary_output=result['binary_output'],
            targets=result['targets'],
            model_name=result['model_name'],
            num_samples=min(4, result['original_output'].shape[0]),
            ts_length=seq_len,
            type_phase=type_phase
        )
        
        if save_plots:
            save_dir = f"./finalize/{output_dir}/{type_phase}/"
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(f"{save_dir}{result['model_name']}_{type_phase}_signals_plot.png", 
                       bbox_inches='tight', dpi=300)
            print(f"Saved plot for {result['model_name']}")
        
        all_figures.append(fig)
        plt.show()
    
    # Comparison plot
    if len(all_outputs_dict) > 1:
        comparison_fig = plot_comparison_summary(
            all_outputs_dict=all_outputs_dict,
            targets=test_targets,
            seq_len=seq_len,
            type_phase=type_phase,
            sample_idx=0
        )
        
        if save_plots:
            save_dir = f"./finalize/{output_dir}/{type_phase}/"
            os.makedirs(save_dir, exist_ok=True)
            comparison_fig.savefig(f"{save_dir}comparison_{type_phase}_plot.png", 
                                 bbox_inches='tight', dpi=300)
            print("Saved comparison plot")
        
        all_figures.append(comparison_fig)
        plt.show()
    
    return all_figures
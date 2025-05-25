import torch
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
import os
from helpers.auxs import AlternatingSignalDataset, generate_inter_trial_interval, split_train_val, split_train_val_og, train, eval, setup_optimizer, visualize_signals
from model_eval import get_model
from pathlib import Path
import datetime
import json
import matplotlib.pyplot as plt


print(os.getcwd())


def trainer(model_name, hidden_s, state_size, memory_size, epochs_length, seed, ckpt_dir, ts_length, lr, clip_grad=None):


    trainset_path = Path(f'/content/mamba-wheels/datasets/train_{ts_length}_colab.pth')
    testset_path = Path(f'/content/mamba-wheels/datasets/test_{ts_length}_colab.pth')
    valset_path = Path(f'/content/mamba-wheels/datasets/val_{ts_length}_colab.pth')
    datasets = [trainset_path, testset_path, valset_path]
    pbar = tqdm(enumerate(datasets))


    # params
    batch_size = 8
    num_workers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    epochs = epochs_length

    
    hidden_size = hidden_s
    state_size = state_size
    
    d_output = 1
    input_size = 1
    output_size = 1
    num_layers = 1
    batch_first = True
    dropout = 0
    prenorm = False

    for _, dataset in pbar:
        np.random.seed(seed)
        print(f'\nMaking a new set of Train, Test and Validation data.')
        hld_ot_int = np.random.randint(0, 99, size=200).tolist()


        if ts_length == 200:
            train_inputs, train_targets, test_inputs, test_targets = generate_inter_trial_interval(num_seq=1000, n=200, iti=(20, 40), isi=10, holdout_intervals=hld_ot_int)
        if ts_length == 600:
            train_inputs, train_targets, test_inputs, test_targets = generate_inter_trial_interval(num_seq=1000, n=600, iti=(60, 120), isi=30 ,holdout_intervals=hld_ot_int)
        if ts_length == 1000:
            train_inputs, train_targets, test_inputs, test_targets = generate_inter_trial_interval(num_seq=1000, n=1000, iti=(100, 200), isi=50 ,holdout_intervals=hld_ot_int)

        train_inputs_split, train_targets_split, val_inputs_split, val_targets_split = split_train_val_og(train_inputs, train_targets, val_split=0.1)


        # Save tensors to a file
        torch.save({'inputs': test_inputs, 'targets': test_targets}, Path(f'datasets/test_{ts_length}.pth'))


        train_set = AlternatingSignalDataset(train_inputs_split, train_targets_split)
        val_set = AlternatingSignalDataset(val_inputs_split, val_targets_split)
        test_set = AlternatingSignalDataset(test_inputs, test_targets)
        # load datasets
        trainset, _ = split_train_val(train_set, val_split=0.2)
        _, valset = split_train_val(val_set, val_split=0.1)
        testset, _ = split_train_val(test_set, val_split=0)
        
        torch.save(trainset, trainset_path)
        torch.save(valset, valset_path)
        torch.save(testset, testset_path)

        print('\nDatasets created!')

        # Dataloaders
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valloader = DataLoader(
            valset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        testloader = DataLoader(
            testset, batch_size=2, shuffle=False, num_workers=num_workers)
        pbar.set_description('Datasets loaded (%d/%d)' % (3,3))

    # save the model params into a file
    with open(f'model_params_experiments_{ts_length}.json', '+a') as file:
        mode_info = {
            'model_name' : model_name +f'_experiments_{ts_length}' ,
            'seed' : seed,
            'hidden_size' : hidden_size,
            'learning_rate' : lr,
            'state/memory_size' : state_size,
            'num_layers' : num_layers,
            'dropout' : dropout,
            'prenorm' : prenorm,
            'batch_size' : batch_size,
            'training_timestamp' : str(datetime.datetime.now()),
            'epochs' : epochs,
            'ts_shape' : trainset.dataset.inputs.shape
        }

        json.dump(mode_info, file, indent=4)

    model = get_model(model_name, input_size, hidden_size, state_size, num_layers)


    if model_name == 'S4D':
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        optimizer, _ = setup_optimizer(model, lr=lr, weight_decay=0.01, epochs=epochs)
    
    else:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)


    model = model.to(device)
    if device == 'cuda':
        cudnn.benchmark = True

    all_mse_list = []
    all_loss_list = []

    print(f'\ntraining {model_name}-{seed}')
    pbar = tqdm(range(0, epochs))
    for epoch in pbar:
        current_date = datetime.date.today()
        formatted_date = current_date.strftime('%Y%m%d')
        pbar.set_description('Epoch: %d' % (epoch))
        # train(model, trainloader, device, optimizer, criterion, epoch, model_name, clip_grad)
        train(model, trainloader, device, optimizer, criterion, epoch, model_name, clip_grad,
          all_mse_list, all_loss_list)
        eval(model, valloader, device, criterion, epoch, model_name, clip_grad)

        if not os.path.isdir(f'./{ckpt_dir}/{model_name}/'):
            os.makedirs(f'./{ckpt_dir}/{model_name}/')
        checkpoint_path = f'./{ckpt_dir}/{model_name}/{model_name}_{lr}_{hidden_size}_{state_size}_{epochs}_ckpt_{seed}_{formatted_date}.pth'

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
    
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(all_mse_list, marker='o')
    axs[0].set_title("MSE over All Batches")
    axs[0].set_xlabel("Batch Index")
    axs[0].set_ylabel("MSE")
    axs[0].grid(True)

    axs[1].plot(all_loss_list, marker='x', color='orange')
    axs[1].set_title("Loss over All Batches")
    axs[1].set_xlabel("Batch Index")
    axs[1].set_ylabel("Loss")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    print(f'\nfinished training {model_name}-{seed}')

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from plot_results import plot_results
from NN_functions_npu import integrate_rk4, loss_rk4, train, train_multi, train_multi_MS, loss_MS
from Neural_architecture import NeuralNetwork


def save_model(model, filename='trained_model.pth'):
    """
    Save the entire model
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model.__class__.__name__
    }, filename)

def save_error(loss_history, filename='loss_history.pth'):
    """
    Save the loss history
    """
    torch.save(loss_history, filename)


def load_model(model_class, layers, filename='trained_model.pth', device='cpu'):
    """
    Load a saved model
    
    Args:
    - model_class: The class of the neural network (e.g., NeuralNetwork)
    - filename: Path to the saved model file
    - device: Device to load the model to
    """
    # Load the checkpoint
    checkpoint = torch.load(filename, map_location=device)
    
    # Recreate the model architecture
    model = model_class(layers)  # Assumes 'layers' is defined globally or passed
    
    # Load the state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to the specified device
    model.to(device)
    
    return model

def load_error(filename='loss_history.pth'):
    """
    Load the loss history
    """
    return torch.load(filename)



def prepare_training(model, data, loss_f, device, iter, lr, batch_size=None):
    """
    Prepares data and model for training
    """
    # Move model to device
    model = model.to(device)
    
    # Prepare data (assuming data is already in the right format)
    A       = data['A']
    C       = data['C']
    t_span  = data['t_span']
    lb      = data['lb']
    ub      = data['ub']
    
    # Train the model

    if batch_size is None:

        trained_model, loss_history = train(
            model, 
            A, 
            C, 
            t_span, 
            lb, 
            ub, 
            device,
            iter,
            lr
        )
    else:
        trained_model, loss_history = train_multi_MS(
            model, 
            A,
            loss_f, 
            C, 
            t_span, 
            lb,
            ub,
            device,
            lr,
            iter,
            batch_size
        )
    
    return trained_model, loss_history

def train_and_save_model(model, data, loss_f, device, iter, lr, batch_size = None, plots = True):
    # Train the model
    trained_model, loss_history = prepare_training(model, data, loss_f, device, iter, lr, batch_size)
    
    # Save the model
    save_model(trained_model, 'trained_metabolic_model.pth')

    # Save the loss history
    save_error(loss_history, 'loss_history.pth')
    
    # Plot results
    if plots:
        if len(data['A']) != 20:
            for i in range(len(data['A'])):
                plot_results(
                    trained_model, 
                    torch.tensor(loss_history), 
                    data['A'][i], 
                    data['C'], 
                    data['t_span'], 
                    nIter=iter,  # Should match training iterations
                    device=device
                )
        else:
            plot_results(
                    trained_model, 
                    torch.tensor(loss_history), 
                    data['A'], 
                    data['C'], 
                    data['t_span'], 
                    nIter=iter,  # Should match training iterations
                    device=device
                )

    
    return trained_model, loss_history

def load_and_plot(layers, train_data, test_data, nIter, filename='trained_metabolic_model.pth', device='cpu', plots = True):
    # Recreate the model architecture
    model = NeuralNetwork(layers).to(device)

    # Load the trained weights
    loaded_model = load_model(NeuralNetwork, layers, filename, device)

    # Load loss error
    loss_history = load_error('loss_history.pth')

    if plots:
        if len(test_data) != 20:
            for i in range(len(train_data['A'])):
                plot_results(
                    loaded_model, 
                    torch.tensor(loss_history), 
                    test_data[i], 
                    train_data['C'], 
                    train_data['t_span'], 
                    nIter=nIter,  # Should match training iterations
                    device=device
                )
        else:
            plot_results(
                    loaded_model, 
                    torch.tensor(loss_history), 
                    test_data, 
                    train_data['C'], 
                    train_data['t_span'], 
                    nIter=nIter,  # Should match training iterations
                    device=device
                )

    # Calculate train loss and test loss
    calculate_MSE(loaded_model, train_data, test_data, device)

    return loaded_model

def calculate_MSE(model, train_data, test_data, device):
    # Calculate train loss and test loss
    train_loss_MS = torch.mean(torch.stack([loss_MS(model, i, train_data['C'], train_data['t_span'], train_data['lb'], train_data['ub'], device) for i in train_data['A']]))
    test_loss_MS  = torch.mean(torch.stack([loss_MS(model, i, train_data['C'], train_data['t_span'], train_data['lb'], train_data['ub'], device) for i in test_data]))
    train_loss = torch.mean(torch.stack([loss_rk4(model, i, train_data['C'], train_data['t_span'], train_data['lb'], train_data['ub'], device) for i in train_data['A']]))
    test_loss  = torch.mean(torch.stack([loss_rk4(model, i, train_data['C'], train_data['t_span'], train_data['lb'], train_data['ub'], device) for i in test_data]))

    # Print results
    print(f"Multistep NN objective value of train data: {train_loss_MS.item()}")
    print(f"Multistep NN objective value of test data: {test_loss_MS.item()}")

    print(f"Train loss: {train_loss.item()}")
    print(f"Test loss: {test_loss.item()}")

    return train_loss, test_loss
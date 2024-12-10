import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# from plot_results import plot_results
# from Neural_architecture import NeuralNetwork

# Integración ODE usando Runge-Kutta 4 (RK4)
def integrate_rk4(model, A_init, C, t_span, device):
    dt = t_span[1] - t_span[0]
    n_steps = len(t_span) - 1
    A_pred = torch.zeros((n_steps + 1, *A_init.shape), device=device)
    A_pred[0] = A_init

    def ode_rhs(A):
        X     = A[-1][-1]
        alpha = model(A)
        dA_dt = torch.matmul(C, alpha.T).T * X
        return dA_dt

    A_curr = A_init.clone()

    for i in range(n_steps):
        k1 = ode_rhs(A_curr)
        k2 = ode_rhs(A_curr + 0.5 * dt * k1)
        k3 = ode_rhs(A_curr + 0.5 * dt * k2)
        k4 = ode_rhs(A_curr + dt * k3)

        A_curr = A_curr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        A_pred[i + 1] = A_curr

    return A_pred  # El resultado ya está como tensor

# Función de pérdida con regularización L2
def loss_rk4(model, A, C, t_span, lb, ub, device):
    A_init = A[0]
    A_init = A_init.unsqueeze(0)
    A_pred = integrate_rk4(model, A_init, C, t_span, device)

    # Escala relativa por variable
    scales = torch.max(torch.abs(A), dim=0)[0]
    scales = torch.where(scales == 0, torch.tensor(1.0, device=device), scales)  # Evitar divisiones por cero

    # Error relativo
    errors = (A_pred - A) / scales
    mse_loss = torch.mean(errors ** 2)  # Error cuadrático medio relativo

    # Error de flujos
    # penalty = loss_flux(model, A, C, lb, ub)
    return mse_loss #+ penalty

def loss_flux(model, A, C, lb, ub):
    # Forward pass
    X     = A[:,-1]
    alphas = model(A/X[:,None])
    
    # Calcular fluxes
    fluxes = torch.einsum('ij,bj->bi', C, alphas)
    
    # Penalización de límites
    penalty_lower = torch.clamp(lb - fluxes, min=0)
    penalty_upper = torch.clamp(fluxes - ub, min=0)
    
    # Normalizar penalty
    scales_f = torch.max(torch.abs(fluxes), dim=0)[0]  # Máximos de cada variable
    scales_f = torch.where(scales_f == 0, torch.tensor(1.0), scales_f)  # Evitar divisiones por cero
    penalty_lower = penalty_lower / scales_f
    penalty_upper = penalty_upper / scales_f
    
    # Calcular la pérdida total
    penalty = torch.mean(penalty_lower**2) + torch.mean(penalty_upper**2)
    
    return penalty


def train(model, A, C, t_span, lb, ub, device, nIter=10000, learning_rate=1e-4):
    # Move data to the specified device
    A  = A.to(device)
    C  = C.to(device)
    lb = lb.to(device)
    ub = ub.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (optional, but can help convergence)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
    
    # Lists to track training progress
    train_loss = []
    
    # Set model to training mode
    model.train()
    
    for it in range(nIter):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Compute loss
        try:
            loss = loss_rk4(model, A, C, t_span, lb, ub, device)
        except Exception as e:
            print(f"Error in loss computation: {e}")
            break
        
        # Compute gradients
        loss.backward()
        
        # Optional: Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        
        # Update weights
        optimizer.step()
        
        # Step the learning rate scheduler
        # scheduler.step()
        
        # Record loss every 100 iterations
        if it % 10 == 0:
            train_loss.append([loss.item()])
            
            # Print loss every 500 iterations
            if it % 10 == 0:
                print(f"Iteration {it}, Loss: {loss.item():.4e}")
    
    return model, train_loss


# # Entrenamiento con múltiples datasets
def train_multi(model, datasets, C, t_span, lb, ub, device, lr=1e-4, n_iter=10000, batch_size=1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
    train_loss = []

    model.eval()
    for it in range(n_iter):
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Calcular pérdida promedio en el batch
            loss = torch.mean(torch.stack([loss_rk4(model, data, C, t_span, lb, ub, device) for data in batch]))
            # Backpropagation
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Actualizar scheduler
            scheduler.step()

        # Guardar pérdida
        if it % 100 == 0:
            train_loss.append(loss.item())
            # if it % 200 == 0:
            print(f"Iteration {it}, Loss: {loss.item():.4e}")

    return model, train_loss


# # Entrenamiento con múltiples datasets
def train_multi(model, datasets, C, t_span, lb, ub, device, lr=1e-4, n_iter=10000, batch_size=1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # model, optimizer = ipex.optimize(model, optimizer=optimizer)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)

    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
    train_loss = []

    model.train()
    for it in range(n_iter):
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Calcular pérdida promedio en el batch
            loss = torch.mean(torch.stack([loss_rk4(model, data, C, t_span, lb, ub, device) for data in batch]))
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Actualizar scheduler
            scheduler.step()

        # Guardar pérdida
        if it % 100 == 0:
            train_loss.append(loss.item())
            print(f"Iteration {it}, Loss: {loss.item():.4e}")

    return model, train_loss

# # Entrenamiento con múltiples datasets
def train_multi_MS(model, datasets, loss_f, C, t_span, lb, ub, device, lr=1e-4, n_iter=10000, batch_size=1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # model, optimizer = ipex.optimize(model, optimizer=optimizer)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)

    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
    train_loss = []

    model.train()
    for it in range(n_iter):
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Calcular pérdida promedio en el batch
            loss = torch.mean(torch.stack([loss_f(model, data, C, t_span, lb, ub, device) for data in batch]))
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Actualizar scheduler
            scheduler.step()

        # Guardar pérdida
        if it % 100 == 0:
            train_loss.append(loss.item())
            print(f"Iteration {it}, Loss: {loss.item():.4e}")

    return model, train_loss

def loss_MS(model, Y, C, t_span, lb, ub, device):
    dt = t_span[1] - t_span[0]
    n_vars = Y.shape[-1]
    Y_normalized = Y / Y[:, -1:].clamp(min=1e-8)  # Evitar divisiones por 0

    # Paso por la red neuronal
    alpha = model(Y_normalized)  # Predicción del lado derecho (timesteps, n_fluxes)

    # Residuo trapezoidal
    Ykp1 = Y[1:, :]    # Y(t+1)
    Yk = Y[:-1, :]     # Y(t)
    alphakp1 = alpha[1:, :]  # f(Y(t+1))
    alphak = alpha[:-1, :]   # f(Y(t))

    # Última columna de Y como X (biomasa)
    Xkp1 = Ykp1[:, -1:]  # X(t+1), shape (timesteps-1, 1)
    Xk = Yk[:, -1:]      # X(t), shape (timesteps-1, 1)

    # Transformar F al espacio de Y usando la matriz C
    fluxkp1 = torch.einsum('ij,tj->ti', C, alphakp1)  # (timesteps-1, n_vars)
    fluxk = torch.einsum('ij,tj->ti', C, alphak)      # (timesteps-1, n_vars)

    # Multiplicar por X en cada paso
    Fkp1_scaled = fluxkp1 * Xkp1  # (timesteps-1, n_vars)
    Fk_scaled = fluxk * Xk        # (timesteps-1, n_vars)

    # Calcular residuos usando el método del trapecio
    res = Ykp1 - Yk - 0.5 * dt * (Fkp1_scaled + Fk_scaled)  # (timesteps-1, n_vars)

    # Normalizar el residuo por el máximo absoluto de cada variable en Y
    max_vals = torch.max(torch.abs(Y), dim=0, keepdim=True).values  # (1, n_vars)
    max_vals = torch.where(max_vals == 0, torch.tensor(1.0, device=device), max_vals)
    res_normalized = res / max_vals
    res_conc = torch.mean(res_normalized ** 2)

    # Concentraciones = calculate_concentrations(Y[0,:], C, alpha, t_span, device)
    # negative_penalty = torch.sum(torch.relu(-Concentraciones))  # Penalización acumulada


    return res_conc  #+  negative_penalty#+ l2_lambda * l2_reg



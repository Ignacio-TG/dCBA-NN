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
        X     = A[-1]
        # alpha = model(A/X)
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

def integrate_rk4_V2(model, A_init, t_span, device):
    dt = t_span[1] - t_span[0]
    n_steps = len(t_span) - 1
    A_pred = torch.zeros((n_steps + 1, *A_init.shape), device=device)
    A_pred[0] = A_init

    def ode_rhs(A):
        return model(A)

    A_curr = A_init.clone()

    for i in range(n_steps):
        k1 = ode_rhs(A_curr)
        k2 = ode_rhs(A_curr + 0.5 * dt * k1)
        k3 = ode_rhs(A_curr + 0.5 * dt * k2)
        k4 = ode_rhs(A_curr + dt * k3)

        A_curr = A_curr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        A_pred[i + 1] = A_curr

    return A_pred  # El resultado ya está como tensor

def integrate_rk4_S(model, A_init, S, t_span, index, device):
    dt = t_span[1] - t_span[0]
    n_steps = len(t_span) - 1
    A_pred = torch.zeros((n_steps + 1, *A_init.shape), device=device)
    A_pred[0] = A_init

    def ode_rhs(A):
        X     = A[-1]
        v = model(A/X)
        v_ex = v[index]
        dA_dt = v_ex * X
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
    A_pred = integrate_rk4(model, A_init, C, t_span, device)

    # Escala relativa por variable
    scales = torch.max(torch.abs(A), dim=0)[0]
    scales = torch.where(scales == 0, torch.tensor(1.0, device=device), scales)  # Evitar divisiones por cero

    # Error relativo
    errors = (A_pred - A) / scales
    mse_loss = torch.mean(errors ** 2)  # Error cuadrático medio relativo

    # Error de flujos
    # penalty = loss_flux(model, A, C, lb, ub)

    return mse_loss

def loss_rk4_V2(model, A, C, t_span, lb, ub, device):
    A_init = A[0]
    A_pred = integrate_rk4_V2(model, A_init, t_span, device)

    # Escala relativa por variable
    scales = torch.max(torch.abs(A), dim=0)[0]
    scales = torch.where(scales == 0, torch.tensor(1.0, device=device), scales)  # Evitar divisiones por cero

    # Error relativo
    errors = (A_pred - A) / scales
    mse_loss = torch.mean(errors ** 2)  # Error cuadrático medio relativo

    # Error de flujos
    # penalty = loss_flux(model, A, C, lb, ub)

    return mse_loss

# Función de pérdida con regularización L2
def loss_rk4_S(model, A, S, t_span, lb, ub, index, device):
    A_init = A[0]
    A_pred = integrate_rk4_S(model, A_init, S, t_span, index, device)

    # Escala relativa por variable
    scales = torch.max(torch.abs(A), dim=0)[0]
    scales = torch.where(scales == 0, torch.tensor(1.0, device=device), scales)  # Evitar divisiones por cero

    # Error relativo
    errors = (A_pred - A) / scales
    mse_loss = torch.mean(errors ** 2)  # Error cuadrático medio relativo

    # Error de flujos
    # penalty = loss_flux(model, A, C, lb, ub)

    return mse_loss

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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
    
    # Lists to track training progress
    train_loss = []
    
    # Set model to training mode
    model.train()
    
    for it in range(nIter):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Compute loss
        try:
            loss = loss_MS(model, A, C, t_span, lb, ub, device)
        except Exception as e:
            print(f"Error in loss computation: {e}")
            break
        
        # Compute gradients
        loss.backward()
        
        # Optional: Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Record loss every 100 iterations
        if it % 100 == 0:
            train_loss.append([loss.item()])
            
            # Print loss every 500 iterations
            if it % 200 == 0:
                print(f"Iteration {it}, Loss: {loss.item():.4e}")
    
    return model, train_loss

# # Entrenamiento con múltiples datasets
def train_multi(model, datasets, C, t_span, lb, ub, device, lr=1e-4, n_iter=10000, batch_size=1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # model, optimizer = ipex.optimize(model, optimizer=optimizer)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8000, gamma=0.1)

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

    # Paso por la red neuronal
    # alpha = model(Y / Y[:, -1:].clamp(min=1e-8))  # Predicción del lado derecho (timesteps, n_fluxes)
    alpha = model(Y)

    # Residuo trapezoidal
    Ykp1     = Y[1:, :]        # Y(t+1)
    Yk       = Y[:-1, :]       # Y(t)
    alphakp1 = alpha[1:, :]    # f(Y(t+1))
    alphak   = alpha[:-1, :]   # f(Y(t))

    # Última columna de Y como X (biomasa)
    Xkp1 = Ykp1[:, -1:]    # X(t+1), shape (timesteps-1, 1)
    Xk   = Yk[:, -1:]      # X(t), shape (timesteps-1, 1)

    # Transformar F al espacio de Y usando la matriz C
    fluxkp1 = torch.einsum('ij,tj->ti', C, alphakp1)  # (timesteps-1, n_vars)
    fluxk   = torch.einsum('ij,tj->ti', C, alphak)      # (timesteps-1, n_vars)

    # Multiplicar por X en cada paso
    Fkp1_scaled = fluxkp1 * Xkp1  # (timesteps-1, n_vars)
    Fk_scaled   = fluxk * Xk        # (timesteps-1, n_vars)

    # Calcular residuos usando el método del trapecio
    res = Ykp1 - Yk - 0.5 * dt * (Fkp1_scaled + Fk_scaled)  # (timesteps-1, n_vars)

    # Normalizar el residuo por el máximo absoluto de cada variable en Y
    max_vals = torch.max(torch.abs(Y), dim=0, keepdim=True).values  # (1, n_vars)
    max_vals = torch.where(max_vals == 0, torch.tensor(1.0, device=device), max_vals)
    res_normalized = res / max_vals

    # Calcular la pérdida ponderada
    res_conc = torch.mean( res_normalized ** 2)

    # # Regularización L1 a los pesos de la red
    # l1_lambda = 0.0001  # Ajusta el valor según sea necesario
    # l1_reg = sum(p.abs().sum() for p in model.parameters())

    # Concentraciones = calculate_concentrations(Y[0,:], C, alpha, t_span, device)
    # negative_penalty = torch.sum(torch.relu(-Concentraciones))  # Penalización acumulada

    return res_conc #+ 0.1*negative_penalty#+ l1_lambda * l1_reg

def loss_simple(model, Y, C, t_span, lb, ub, device):
    dt = t_span[1] - t_span[0]

    # Paso por la red neuronal
    # alpha = model(Y / Y[:, -1:].clamp(min=1e-8))  # Predicción del lado derecho (timesteps, n_fluxes)
    f = model(Y)

    # Residuo trapezoidal
    Ykp1     = Y[1:, :]        # Y(t+1)
    Yk       = Y[:-1, :]       # Y(t)
    fkp1     = f[1:, :]    # f(Y(t+1))
    fk       = f[:-1, :]   # f(Y(t))

    
    # Calcular residuos usando el método del trapecio
    res = Ykp1 - Yk - 0.5 * dt * (fkp1 + fk)  # (timesteps-1, n_vars)

    # Normalizar el residuo por el máximo absoluto de cada variable en Y
    max_vals = torch.max(torch.abs(Y), dim=0, keepdim=True).values  # (1, n_vars)
    max_vals = torch.where(max_vals == 0, torch.tensor(1.0, device=device), max_vals)
    res_normalized = res / max_vals

    # Calcular la pérdida ponderada
    res_conc = torch.mean( res_normalized ** 2)

    # # Regularización L1 a los pesos de la red
    # l1_lambda = 0.0001  # Ajusta el valor según sea necesario
    # l1_reg = sum(p.abs().sum() for p in model.parameters())

    # Concentraciones = calculate_concentrations(Y[0,:], C, alpha, t_span, device)
    # negative_penalty = torch.sum(torch.relu(-Concentraciones))  # Penalización acumulada

    return res_conc #+ 0.1*negative_penalty#+ l1_lambda * l1_reg

def calculate_concentrations(Y0, C, alpha_matrix, t_span, device):

    dt = t_span[1] - t_span[0]
    
    # Normalizar la concentración inicial
    Concentracion_t = Y0 
    Concentraciones = [Concentracion_t]  # Almacenar concentraciones calculadas

    # Iterar para calcular cada paso en el tiempo
    for t in range(1, len(t_span)):
        # Usar alpha precomputado para los tiempos t y t+1
        alpha_t = alpha_matrix[t - 1]  # Predicción precomputada (shape [n_fluxes])
        alpha_tp1 = alpha_matrix[t] if t < len(alpha_matrix) else alpha_matrix[t - 1]

        # Transformar alpha al espacio de las concentraciones usando la matriz C
        flux_t = torch.einsum('ij,j->i', C, alpha_t)      # Flujo en t
        flux_tp1 = torch.einsum('ij,j->i', C, alpha_tp1)  # Flujo en t+1

        # Escalar por la biomasa actual
        F_t_scaled = flux_t * Concentracion_t[-1]      # Flujo escalado en t
        F_tp1_scaled = flux_tp1 * Concentracion_t[-1]  # Flujo escalado en t+1

        # Cálculo recursivo de la concentración en el siguiente paso
        Concentracion_t1 = Concentracion_t + 0.5 * dt * (F_tp1_scaled - F_t_scaled)  # Método del trapecio

        # Almacenar y actualizar para el siguiente paso
        Concentraciones.append(Concentracion_t1)
        Concentracion_t = Concentracion_t1  # Actualización

    # Convertir las concentraciones calculadas en un tensor
    return torch.stack(Concentraciones)  # (timesteps, n_vars)

def loss_MS_Sv(model, Y, S, t_span, lb, ub, index, device):
    dt = t_span[1] - t_span[0]
    n_vars = Y.shape[-1]
    Y_normalized = Y / Y[:, -1:].clamp(min=1e-8)  # Evitar divisiones por 0

    # Paso por la red neuronal
    v = model(Y_normalized)  # Predicción del lado derecho (timesteps, n_fluxes)
    v_ex = v[:, index]

    # Residuo trapezoidal
    Ykp1     = Y[1:, :]    # Y(t+1)
    Yk       = Y[:-1, :]     # Y(t)
    vkp1     = v_ex[1:, :]  # f(Y(t+1))
    vk       = v_ex[:-1, :]   # f(Y(t))

    # Última columna de Y como X (biomasa)
    Xkp1 = Ykp1[:, -1:]  # X(t+1), shape (timesteps-1, 1)
    Xk = Yk[:, -1:]      # X(t), shape (timesteps-1, 1)

    # Multiplicar por X en cada paso
    Fkp1_scaled = vkp1 * Xkp1  # (timesteps-1, n_vars)
    Fk_scaled   = vk * Xk        # (timesteps-1, n_vars)

    # Calcular residuos usando el método del trapecio
    res = Ykp1 - Yk - 0.5 * dt * (Fkp1_scaled + Fk_scaled)  # (timesteps-1, n_vars)

    # Normalizar el residuo por el máximo absoluto de cada variable en Y
    max_vals = torch.max(torch.abs(Y), dim=0, keepdim=True).values  # (1, n_vars)
    max_vals = torch.where(max_vals == 0, torch.tensor(1.0, device=device), max_vals)
    res_normalized = res / max_vals
    res_conc = torch.mean(res_normalized ** 2)


    # Concentraciones = calculate_concentrations(Y[0,:], C, alpha, t_span, device)
    # negative_penalty = torch.sum(torch.relu(-Concentraciones))  # Penalización acumulada
    products = torch.stack([torch.matmul(S, v_i) for v_i in v])
    penalty = torch.sum(products ** 2)

    # print(products ** 2)
    # print(penalty)

    return res_conc + 0.01*penalty #+  negative_penalty#+ l2_lambda * l2_reg

def train_multi_MS_S(model, datasets, loss_f, C, t_span, lb, ub, index, device, lr=1e-4, n_iter=10000, batch_size=1):
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
            loss = torch.mean(torch.stack([loss_f(model, data, C, t_span, lb, ub, index, device) for data in batch]))
            
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
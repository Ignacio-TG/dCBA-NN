import numpy as np
import matplotlib.pyplot as plt
import torch

from NN_functions_npu import train, integrate_rk4

def plot_results(trained_model, trained_error, A, C, t_span, nIter, device):
    # Ensure everything is on the right device and converted to numpy for plotting
    A_np = A.cpu().numpy()
    C_np = C.cpu().numpy()
    t_span_np = t_span.cpu().numpy()

    # Disable gradient computation for inference
    with torch.no_grad():
        # Move model to CPU for plotting
        trained_model.to('cpu')
        
        # Simulate using the trained model
        A_init = A[0]
        A_sim = integrate_rk4(trained_model, A_init, C, t_span, device).cpu().numpy()

    # Plotting setup
    plt.figure(figsize=(15, 10))

    # Subplot titles and indices (glc, ace, o2, co2, nh4, bio)
    subplot_configs = [
        (0, 'Glucosa'),
        (1, 'Acetato'),
        (2, 'Oxígeno'),
        (3, 'CO2'),
        (4, 'Amonio'),
        (5, 'Biomasa')
    ]

    # Create subplots for each variable
    for i, (idx, title) in enumerate(subplot_configs, 1):
        plt.subplot(3, 2, i)
        plt.plot(t_span_np, A_sim[:, idx], label='Simulación')
        plt.plot(t_span_np, A_np[:, idx], 'o', label='Experimental')
        plt.xlabel('Tiempo (h)')
        plt.ylabel('Concentración (g/L)')
        plt.title(title)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

    # Glucose Flux Plot
    v_max = 10
    K_S = 0.5
    flux_exp = (A_np[:, 0]) * v_max / (K_S + A_np[:, 0])
    flux_sim = (A_sim[:, 0]) * v_max / (K_S + A_sim[:, 0])
    
    plt.figure()
    plt.plot(t_span_np, flux_exp,'o', label='Experimental')
    plt.plot(t_span_np, flux_sim, label='Simulación')
    plt.legend()
    plt.xlabel('Tiempo (h)')
    plt.ylabel('Flujo (g/gDW h)')
    plt.title('Flujo de glucosa')
    plt.grid(True)
    plt.show()

    # Training Error Plot
    plt.figure(figsize=(10, 6))
    
    # Convert trained_error to numpy if it's a torch tensor
    if torch.is_tensor(trained_error):
        trained_error = trained_error.cpu().numpy()

    plt.plot(np.linspace(0, nIter, len(trained_error)), trained_error)
    plt.xlabel('Iteración')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.title('Error de entrenamiento')
    plt.grid(True)
    plt.show()

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# Import functions  class
from NN_functions import train, integrate_rk4, loss_rk4, loss_MS, loss_simple, integrate_rk4_V2
from plot_results import plot_results, plot_results_simple, plot_comparison
from Neural_architecture import NeuralNetwork
from Train_save_load import save_model, load_model, prepare_training, train_and_save_model, load_and_plot, load_error, calculate_MSE, calculate_MSE_simple

# Importar matriz espacio nulo
N_ex = pd.read_csv('Matrices/N_ex.csv', header=None)

# Eliminar primera fila de N_ex y convertir a tensor de torch
N_ex = torch.tensor(N_ex.iloc[1:, :].values.astype(np.float32))

# Importar indices de reacciones importantes
index_path  = 'Matrices/Indices_N_ex.csv'
index       = pd.read_csv(index_path, dtype=int)
bio_idx     = int(index['bio_met_dcba'].iloc[0]) - 1
glc_idx     = int(index['glc_met_dcba'].iloc[0]) - 1
ac_idx      = int(index['ac_met_dcba'].iloc[0]) - 1
o2_idx      = int(index['o2_met_dcba'].iloc[0]) - 1
co2_idx     = int(index['co2_met_dcba'].iloc[0]) - 1
nh4_idx     = int(index['nh4_met_dcba'].iloc[0]) - 1
etoh_idx    = int(index['etoh_met_dcba'].iloc[0]) - 1

## Importar datos experimentales
data_list = []
data_norm = []
path      = ['Data Aerobica/Aerobico_' + str(i) + '.csv' for i in range(1,23)]

for file in path:
    # Carga el archivo CSV
    data = pd.read_csv(file)

    # Extrae columnas y convierte a tensores de torch
    tiempo  = torch.tensor(data["Time"].values[::3], dtype=torch.float32)
    bio     = torch.tensor(data["X"].values[::3], dtype=torch.float32)
    glc     = torch.tensor(data["Glc"].values[::3], dtype=torch.float32)
    ace     = torch.tensor(data["Ace"].values[::3], dtype=torch.float32)
    o2      = torch.tensor(data["O2"].values[::3], dtype=torch.float32)
    co2     = torch.tensor(data["CO2"].values[::3], dtype=torch.float32)
    nh4     = torch.tensor(data["NH4"].values[::3], dtype=torch.float32)

    # Combina las variables en un solo tensor:
    Y = torch.stack([glc, ace, o2, co2, nh4, bio], dim=1)

    # Normaliza cada columna por bio, excepto bio
    Y_normalized = Y[:, :-1] / Y[:, -1:].clamp(min=1e-8)
    Y_normalized = torch.cat((Y_normalized, Y[:, -1:]), dim=1)

    # Guarda los datos normalizados en una lista
    data_norm.append(Y_normalized)

    # Guarda los datos en una lista
    data_list.append(Y)

# Convertir límites a tensores de torch
lb = torch.tensor([-10, -53.6, -10, -11.2, -15.5, 0], dtype=torch.float32)
ub = torch.tensor([0, 20, 0, 62.4, 0, 0.6], dtype=torch.float32)

# Inicialización
device = torch.device('cpu')

n_metabolites = 6
# layers = [n_metabolites,100, 100, 10, 100, 100, 24]
layers = [n_metabolites, 200, 200, 200, 200, 200, 24]
model = NeuralNetwork(layers).to(device)

C      = N_ex[[glc_idx, ac_idx, o2_idx, co2_idx, nh4_idx, bio_idx], :].clone().detach().to(device)
t_span = tiempo.clone().detach().to(device)

# Convertir datasets a tensores de PyTorch
# datasets = [data.clone().detach().to(device) for data in data_list]
datos_curados = [0, 3, 5, 6, 11, 13, 14, 16, 17]
# datos_curados = [3, 5, 6, 13, 14]

datasets = [data_list[i].clone().detach().to(device) for i in datos_curados]

num_datasets = len(datasets)
num_train = int(0.6 * num_datasets)  # Número de datasets para entrenamiento (70%)

# Obtener índices aleatorios para el set de entrenamiento
train_indices = random.sample(range(num_datasets), num_train)

# Crear sets de entrenamiento y testeo
train_data = [datasets[i] for i in train_indices]
test_data = [datasets[i] for i in range(num_datasets) if i not in train_indices]

# Prepare your data
data = {
    'A': datasets[0],
    'C': C,
    't_span': t_span,
    'lb': lb,
    'ub': ub,
}

# model = ipex.optimize(model)
# torch.autograd.set_detect_anomaly(True)
# Train and save
# trained_model, loss_history = train_and_save_model(model, data, loss_MS, device, 20000, 1e-3, batch_size=None, plots=False)

# Plot using validation data

# load_and_plot(layers, data, test_data, 15000, 'trained_metabolic_model.pth', 'loss_history.pth', device, True)

# load_and_plot(layers, data, test_data, 20000, 'Entrenamientos/trained_metabolic_model_10.pth', 'Entrenamientos/loss_history_10.pth', device, False)


# Datos a utilizar
# 0, 3, 5, 6, 8, 11, 13, 14, 16, 17
# 9,15, 20, 21, diauxico god

trained_model = load_model(NeuralNetwork, [6, 200, 200, 200, 200, 200, 24], 'Entrenamientos/trained_metabolic_model_over.pth', device)
trained_error = load_error('Entrenamientos/loss_history_over.pth')
plot_results(trained_model, trained_error, datasets[0], C, t_span, 30000, device)

# # calculate_MSE_simple(trained_model, data, test_data, device)

# trained_model_simple = load_model(NeuralNetwork, [6, 400, 400, 400, 6], 'Entrenamientos/trained_metabolic_model_simple.pth', device)
# trained_model_dcba   = load_model(NeuralNetwork, [6, 200, 200, 200, 200, 200, 24], 'Entrenamientos/trained_metabolic_model_over.pth', device)

# # plot_comparison(trained_model_dcba, trained_model_simple, datasets[0], C, t_span, device)


# # Comparación tiempos de ejecución
# import time

# t_simple = []
# t_dcba   = []

# for i in range(100):
#     # Simulate using the trained model
#     A_init = datasets[0][0]
#     start = time.time()
#     A_alpha = integrate_rk4(trained_model_dcba, A_init, C, t_span, device)
#     end = time.time()
#     t_simple.append(end - start)

    
#     start = time.time()       
#     A_simple = integrate_rk4_V2(trained_model_simple, A_init, t_span, device)
#     end = time.time()
#     t_dcba.append(end - start)

# print('NeuralODE:', np.mean(t_simple))
# print('dCBA-NN:', np.mean(t_dcba))


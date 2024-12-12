
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# Import functions  class
from NN_functions import train, integrate_rk4, loss_rk4, loss_MS, loss_simple, integrate_rk4_V2
from Funciones_auxiliares.plot_results import plot_results, plot_results_simple, plot_comparison, plot_results_conv
from Neural_architecture import NeuralNetwork
from Funciones_auxiliares.Train_save_load import save_model, load_model, prepare_training, train_and_save_model, load_and_plot, load_error, calculate_MSE, calculate_MSE_simple, save_error
from NN_convolucional import ConvNeuralNetwork, train_multi_MS_Conv, loss_MS_Conv

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
    tiempo  = torch.tensor(data["Time"].values[::4], dtype=torch.float32)
    bio     = torch.tensor(data["X"].values[::4], dtype=torch.float32)
    glc     = torch.tensor(data["Glc"].values[::4], dtype=torch.float32)
    ace     = torch.tensor(data["Ace"].values[::4], dtype=torch.float32)
    o2      = torch.tensor(data["O2"].values[::4], dtype=torch.float32)
    co2     = torch.tensor(data["CO2"].values[::4], dtype=torch.float32)
    nh4     = torch.tensor(data["NH4"].values[::4], dtype=torch.float32)

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
conv_filters = [8, 16, 16, 8]  # Ejemplo de filtros para las capas convolucionales
dense_layers = [24, 24, 24, 24]  # Ejemplo de neuronas para las capas densas
model = ConvNeuralNetwork((conv_filters, dense_layers))


C = N_ex[[glc_idx, ac_idx, o2_idx, co2_idx, nh4_idx, bio_idx], :].clone().detach().to(device)
t_span = tiempo.clone().detach().to(device)

datos_curados = [0, 3, 5, 6, 11, 13, 14, 16, 17]
datasets = [data_list[i].clone().detach().to(device) for i in datos_curados]

num_datasets = len(datasets)
num_train = int(0.6 * num_datasets)
train_indices = random.sample(range(num_datasets), num_train)
train_data = [datasets[i] for i in train_indices]
test_data = [datasets[i] for i in range(num_datasets) if i not in train_indices]

data = {
    'A': datasets[0],
    'C': C,
    't_span': t_span,
    'lb': lb,
    'ub': ub,
}


# trained_model, trained_error =  train_multi_MS_Conv(model, datasets, loss_MS_Conv, C, t_span, lb, ub, device, lr=1e-3, n_iter=10000, batch_size=3)

# save_model(trained_model, 'Entrenamientos/trained_metabolic_model_conv.pth')
# save_error(trained_error, 'Entrenamientos/loss_history_conv.pth')
# Cargar y plotear resultados con ConvNN

trained_model = load_model(ConvNeuralNetwork, (conv_filters, dense_layers), 'Entrenamientos/trained_metabolic_model_conv.pth', device)
trained_error = load_error('Entrenamientos/loss_history_conv.pth')
plot_results_conv(trained_model, trained_error, train_data[2], C, t_span, 10000, device)
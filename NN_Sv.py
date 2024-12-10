
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# Import functions  class
from NN_functions import train, integrate_rk4, loss_rk4, loss_MS, loss_MS_Sv
from plot_results import plot_results, plot_results_S
from Neural_architecture import NeuralNetwork
from Train_save_load import save_model, load_model, prepare_training, train_and_save_model_S, load_and_plot, load_and_plot_S

# Importar matriz espacio nulo
N_ex_path = 'Matrices/N_ex.csv'
N_ex = pd.read_csv(N_ex_path, header=None)

# Eliminar primera fila de N_ex y convertir a tensor de torch
N_ex = torch.tensor(N_ex.iloc[1:, :].values.astype(np.float32))

#Importar matriz estequiométrica y límites superiores e inferiores
S    = pd.read_csv('Matrices/S.csv', header=None)
S    = torch.tensor(S.values.astype(np.float32))

lb_v = pd.read_csv('Matrices/lb.csv', header=None)
lb_v = torch.tensor(lb_v.values.astype(np.float32))

ub_v = pd.read_csv('Matrices/ub.csv', header=None)
ub_v = torch.tensor(ub_v.values.astype(np.float32))

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

# Ordenar índices para entrenar
index = torch.tensor([glc_idx, ac_idx, o2_idx, co2_idx, nh4_idx, bio_idx])

## Importar datos experimentales
data_list = []
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

    # Guarda los datos en una lista
    data_list.append(Y)

# Convertir límites a tensores de torch
lb = torch.tensor([-10, -53.6, -10, -11.2, -15.5, 0], dtype=torch.float32)
ub = torch.tensor([0, 20, 0, 62.4, 0, 0.6], dtype=torch.float32)

# Inicialización
device = torch.device('cpu')

n_metabolites = 6
layers = [n_metabolites,200, 200, 200, 200, 96]

model = NeuralNetwork(layers).to(device)

C      = N_ex[[glc_idx, ac_idx, o2_idx, co2_idx, nh4_idx, bio_idx], :].clone().detach().to(device)
t_span = tiempo.clone().detach().to(device)

# Convertir datasets a tensores de PyTorch
datasets = [data.clone().detach().to(device) for data in data_list]

# # Dividir datasets en set de entrenamiento y testeo
# train_data = [datasets[2], datasets[3], datasets[8], datasets[12], datasets[0], datasets[20]]
# # train_data = datasets[0]
# test_data  = [datasets[15], datasets[1], datasets[10]]

num_datasets = len(datasets)
num_train = int(0.6 * num_datasets)  # Número de datasets para entrenamiento (70%)

# Obtener índices aleatorios para el set de entrenamiento
train_indices = random.sample(range(num_datasets), num_train)

# Crear sets de entrenamiento y testeo
train_data = [datasets[i] for i in train_indices]
test_data = [datasets[i] for i in range(num_datasets) if i not in train_indices]

# Prepare your data
data = {
    'A': train_data,
    'S': S,
    't_span': t_span,
    'lb': lb,
    'ub': ub,
    'index': index
}

# model = ipex.optimize(model)

# Train and save
trained_model, loss_history = train_and_save_model_S(model, data, loss_MS_Sv, device, 20000, 1e-3, batch_size=2, plots=False)

# Plot using validation data
# load_and_plot_S(layers, data, test_data, 20000, 'trained_metabolic_model.pth', device, True)

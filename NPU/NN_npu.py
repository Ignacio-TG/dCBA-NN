import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import intel_npu_acceleration_library
from intel_npu_acceleration_library.compiler import CompilerConfig
# from intel_npu_acceleration_library import NPUDevice


# Import functions  class
from NN_functions_npu import train, integrate_rk4, loss_rk4, loss_MS
from plot_results import plot_results
from Neural_architecture import NeuralNetwork
from Train_save_load_npu import save_model, load_model, prepare_training, train_and_save_model, load_and_plot

# Importar matriz espacio nulo
N_ex_path = '../Matrices/N_ex.csv'
N_ex = pd.read_csv(N_ex_path, header=None)

# Eliminar primera fila de N_ex y convertir a tensor de torch
N_ex = torch.tensor(N_ex.iloc[1:, :].values.astype(np.float32))

# Importar indices de reacciones importantes
index_path  = '../Matrices/Indices_N_ex.csv'
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
path      = ['../Data Aerobica/Aerobico_' + str(i) + '.csv' for i in range(1,23)]

for file in path:
    # Carga el archivo CSV
    data = pd.read_csv(file)

    # Extrae columnas y convierte a tensores de torch
    tiempo  = torch.tensor(data["Time"].values[::5], dtype=torch.float32)
    bio     = torch.tensor(data["X"].values[::5], dtype=torch.float32)
    glc     = torch.tensor(data["Glc"].values[::5], dtype=torch.float32)
    ace     = torch.tensor(data["Ace"].values[::5], dtype=torch.float32)
    o2      = torch.tensor(data["O2"].values[::5], dtype=torch.float32)
    co2     = torch.tensor(data["CO2"].values[::5], dtype=torch.float32)
    nh4     = torch.tensor(data["NH4"].values[::5], dtype=torch.float32)

    # Combina las variables en un solo tensor:
    Y = torch.stack([glc, ace, o2, co2, nh4, bio], dim=1)

    # Guarda los datos en una lista
    data_list.append(Y)


# Convertir límites a tensores de torch
lb = torch.tensor([-10, -53.6, -10, -11.2, -15.5, 0], dtype=torch.float32)
ub = torch.tensor([0, 20, 0, 62.4, 0, 0.6], dtype=torch.float32)

# Inicialización
device = torch.device("npu")
# device = NPUDevice()

n_metabolites = 6
layers = [n_metabolites, 100, 100, 20, 100, 100, 24]
model = NeuralNetwork(layers)

compiler_conf = CompilerConfig(dtype=torch.float32, training=True)
model = intel_npu_acceleration_library.compile(model, compiler_conf)

C      = torch.tensor(N_ex[[glc_idx, ac_idx, o2_idx, co2_idx, nh4_idx, bio_idx], :], dtype=torch.float32, device=device)
t_span = torch.tensor(tiempo, dtype=torch.float32, device=device)

# Convertir datasets a tensores de PyTorch
datasets = [torch.tensor(data, dtype=torch.float32, device=device) for data in data_list]

# Dividir datasets en set de entrenamiento y testeo
train_data = [datasets[2], datasets[3], datasets[8]]
test_data  = datasets[5:8]

# Prepare your data
data = {
    'A': train_data,
    'C': C,
    't_span': t_span,
    'lb': lb,
    'ub': ub
}

# Train and save
trained_model, loss_history = train_and_save_model(model, data, loss_MS, device, 20000, 1e-3, batch_size=3, plots=True)

# # Plot using validation data
# data['A'] = test_data
# load_and_plot(layers, data, 3000, 'trained_metabolic_model.pth', device)

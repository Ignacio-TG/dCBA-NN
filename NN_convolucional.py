import torch
from torch import nn, optim
from torch.utils.data import DataLoader

class ConvNeuralNetwork(nn.Module):

    def __init__(self, layers):
        super(ConvNeuralNetwork, self).__init__()
        conv_filters, dense_units = layers
        # Capas convolucionales
        self.conv_layers = nn.Sequential(
            # Primera convolución: filtro 6x1 para procesar columnas individualmente
            nn.Conv2d(in_channels=1, out_channels=conv_filters[0], kernel_size=(6, 1), stride=1, padding=0),
            nn.BatchNorm2d(conv_filters[0]),
            nn.Tanh(),

            # Segunda convolución: extraer características cruzadas con filtro 1x3
            nn.Conv2d(in_channels=conv_filters[0], out_channels=conv_filters[1], kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(conv_filters[1]),
            nn.Tanh(),

            # Ajuste del MaxPooling: reduce columnas pero preserva al menos una dimensión válida
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            # Tercera convolución: procesar características con un filtro 3x3
            nn.Conv2d(in_channels=conv_filters[1], out_channels=conv_filters[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_filters[2]),
            nn.Tanh(),

            # Cuarta convolución: más extracción de características con un filtro 3x3
            nn.Conv2d(in_channels=conv_filters[2], out_channels=conv_filters[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_filters[3]),
            nn.Tanh(),

            # Ajuste del MaxPooling: garantizar que las dimensiones no se reduzcan a 0
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),

            # Regularización con Dropout
            # nn.Dropout(0.1)
        )

        # Calcular el tamaño aplanado después de las capas convolucionales
        # Las dimensiones de salida deben calcularse explícitamente
        sample_input = torch.zeros(1, 1, 6, 6)  # Tamaño de entrada esperado
        output_shape = self.conv_layers(sample_input).shape  # Dimensión de salida después de convoluciones
        flattened_size = output_shape[1] * output_shape[2] * output_shape[3]

        # Capas densas
        self.dense_layers = nn.ModuleList()
        self.dense_layers.append(nn.Linear(flattened_size, dense_units[0]))

        for i in range(len(dense_units) - 1):
            self.dense_layers.append(nn.Linear(dense_units[i], dense_units[i + 1]))

        self.activation = nn.Tanh()

    def forward(self, x):
        # Transformar el vector de entrada en una matriz 6x6 replicando el vector
        # Asume que el input tiene forma [batch_size, 6] (1 fila, 6 columnas por batch)
        x = x.unsqueeze(2).repeat(1, 1, 6)  # Expande la dimensión y repite por columnas
        x = x.permute(0, 2, 1).unsqueeze(1)  # Reordena a [batch_size, 1, 6, 6]
        
        # Paso por capas convolucionales
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Aplanar

        # Paso por capas densas
        for layer in self.dense_layers[:-1]:
            x = self.activation(layer(x))
        x = self.dense_layers[-1](x)  # Última capa sin activación
        return x


# Entrenamiento con múltiples datasets
def train_multi_MS_Conv(model, datasets, loss_f, C, t_span, lb, ub, device, lr=1e-4, n_iter=10000, batch_size=1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
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

def loss_MS_Conv(model, Y, C, t_span, lb, ub, device):
    dt = t_span[1] - t_span[0]

    # Paso por la red neuronal
    # Y_reshaped = Y.view(Y.size(0), 1, 6, 6)  # Convertir entrada a matriz 6x6
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
    res_conc = torch.mean(res_normalized ** 2)

    return res_conc

def integrate_rk4_conv(model, A_init, C, t_span, device):
    dt = t_span[1] - t_span[0]
    n_steps = len(t_span) - 1
    A_pred = torch.zeros((n_steps + 1, *A_init.shape), device=device)
    A_pred[0] = A_init

    def ode_rhs(A):
        X = A[-1]
        # Asegúrate de que A tenga las dimensiones correctas
        A_input = A.view(1, -1)  # Transformar A en un tensor de tamaño [1, 6]
        alpha = model(A_input)
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
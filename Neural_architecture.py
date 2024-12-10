import torch
from torch import nn

class NeuralNetwork(nn.Module):
  def __init__(self, layers):
    super(NeuralNetwork, self).__init__()
    self.layers = nn.ModuleList()
    for i in range(len(layers) - 1):
      self.layers.append(nn.Linear(layers[i], layers[i + 1]))
    self.activation = nn.Tanh()


  def forward(self, x):
    for layer in self.layers[:-1]:
      x = self.activation(layer(x))
    x = self.layers[-1](x)  # Última capa sin activación
    return x
  

  
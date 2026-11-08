import torch.nn as nn

class RedNeuronalSimple(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RedNeuronalSimple, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Función de activación ReLU (comúnmente usada en PyTorch)
        self.relu = nn.ReLU()

        # Capa de salida
        # No se aplica activación en la salida para problemas de regresión estándar.
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Aseguramos que la entrada sea de tipo float
        x = x.float()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
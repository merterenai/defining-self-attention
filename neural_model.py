from self_attention import self_attention
import random
import math
from self_attention import matmul

def relu(matrix):
    output = []
    for row in matrix:
        for i in range(len(row)):
            row[i] = max(0,row[i])
        output.append(row)
    return output   
def rand(*size):
    """
    PyTorch'taki torch.rand() fonksiyonunun saf Python karşılığı.
    *size: Boyutları temsil eden argümanlar (örneğin: 2, 3)
    """
    if len(size) == 1:
        # Tek boyutlu liste
        return [random.random() for _ in range(size[0])]
    elif len(size) == 2:
        # İki boyutlu matris
        return [[random.random() for _ in range(size[1])] for _ in range(size[0])]

class NeuralModel:
    def __init__(self,hidden_size,output_size):
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embeddings = rand(hidden_size, hidden_size)
        self.W_in = rand(hidden_size, hidden_size)
        self.W_out = rand(hidden_size, output_size)
        self.self_attention = self_attention

    def forward(self, input_sequence):
        embeddings = [self.embeddings[i] for i in input_sequence]
        W_in = matmul(embeddings, self.W_in)
        W_in = relu(W_in)
        attention_output = self.self_attention(W_in, W_in, W_in)
        output = matmul(attention_output, self.W_out)
        return output
    

if __name__ == "__main__":
    model = NeuralModel(hidden_size=4, output_size=2)
    input_sequence = [0, 1, 2, 3]
    output = model.forward(input_sequence)
    print("Model Output:")
    print(output)
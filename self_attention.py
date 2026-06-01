import math

K = []
V = []
Q = []


def dot_product(K,Q):
    return sum(K*Q for K, Q in zip(K,Q))

def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def matmul(A, B):
    """İki matrisi çarpar (A @ B)."""
    # B matrisinin transpozunu alarak sütunlara erişimi kolaylaştırıyoruz
    B_T = transpose(B)
    return [[sum(a * b for a, b in zip(row, col)) for col in B_T] for row in A]

def softmax(row):
    exp_x = [math.exp(x) for x in row]
    sum_exp_x = sum(exp_x)
    return [x / sum_exp_x for x in exp_x]

def self_attention(K,Q,V):
    d_k = len(K[0])
    K_T = transpose(K)
    attention_scores = []
    for i in range(len(Q)):
        scores = []
        for j in range(len(K_T)):
            score = dot_product(K_T[j], Q[i]) / math.sqrt(d_k)
            scores.append(score)
        
        weights = softmax(scores)

        output_row =  [0] * len(V[0])
        for i in range(len(weights)):

            for k in range(len(V[0])):

                output_row[k] += weights[i] * V[i][k]
        attention_scores.append(output_row)
    return attention_scores



if __name__ == "__main__":
    K = [[1, 0], [0, 1]]
    V = [[1, 2], [3, 4]]
    Q = [[1, 0], [0, 1]]
    output = self_attention(K, Q, V)
    print("Attention Output:")
    print(output)

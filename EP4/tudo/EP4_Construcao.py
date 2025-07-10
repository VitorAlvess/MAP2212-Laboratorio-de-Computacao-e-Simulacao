import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt 
# Parâmetros de entrada
x = [4, 6, 4]
y = [1, 2, 3]
n = 15366400  # Número de amostras (Vamos descrever como calcular isso no relatorio)
k = 4000     # Número de bins (cortes) (Vamos descrever como calcular isso no relatorio)

# 1️⃣ Gerar o vetor α somando x + y
alpha = np.array(x) + np.array(y)

# 2️⃣ Calcular a constante de normalização (Beta Multivariada)
beta = (np.prod(gamma(alpha))) / (gamma(np.sum(alpha)))
const_norm = 1 / beta

print(f"Constante de normalização calculada: {const_norm:.6f}")

# 3️⃣ Gerar as amostras de theta usando a distribuição Dirichlet
theta = np.random.dirichlet(alpha, n)

# 4️⃣ Calcular f(θ) para cada amostra gerada
f_theta = np.prod(np.power(theta, alpha - 1), axis=1)

# 5️⃣ Normalizar os valores de f(θ)
f_normalizada = f_theta * const_norm

# 6️⃣ Ordenar os valores para facilitar a separação em bins
f_normalizada.sort()

print(f"Primeiros 5 valores normalizados e ordenados: {f_normalizada[:5]}")

# 7️⃣ Separar os valores em bins com quantidade constante de pontos
f_bins = [0] * k
passo = int(n / k)

for i in range(k):
    f_bins[i] = f_normalizada[i * passo]

print(f"Primeiros 5 bins calculados: {f_bins[:5]}")


def U(v, f_bins, k):
    """
    Calcula U(v) a partir dos bins.
    
    Parâmetros:
    - v: valor de corte
    - f_bins: os valores normalizados separados em bins
    - k: número de bins
    
    Retorno:
    - u: valor acumulado em relação a v
    """
    f_min = f_bins[0]
    f_max = f_bins[-1]

    if v > f_max:
        return 1.0
    if v < f_min:
        return 0.0

    # Busca binária para encontrar onde está o valor
    menor_que_v = np.searchsorted(f_bins, v, side='left')
    u = menor_que_v / k

    return u



v = np.linspace(f_bins[0] - 1, f_bins[-1] + 1, 1000)
probabilidade_acumulada = [U(valor, f_bins, k) for valor in v]

def plotar_grafico(v, probabilidade_acumulada):
    #Utilizado ferramentas de copilot para gerar o gráfico
    plt.figure(figsize=(8, 5))
    plt.plot(v, probabilidade_acumulada, label='U(v)')
    plt.title("Função Acumulada U(v)")
    plt.xlabel("v")
    plt.ylabel("Probabilidade Acumulada")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
plotar_grafico(v, probabilidade_acumulada)



for v in range(20):
    print(f"U({v}) = {U(v, f_bins, k):.6f}")
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt 
import time

#Nomes e números USP
# José Victor Santos Alves - 14713085
# Marcos Elias Jara Grubert - 1295930
# Parâmetros de Entrada:
inicio = time.time()
x = [4, 6, 4]
y = [1, 2, 3]
n = 15366400  # Quantidade de amostras a serem geradas (Explicado no Relatório o calculo)
k = 2000      # Número de regiões (bins) (Explicado no Relatório o calculo)


# Passo 1: Definindo os Parâmetros da Dirichlet

# Somando os vetores x e y para gerar o vetor de parâmetros
parametros = np.array(x) + np.array(y)


# Passo 2: Calculando a Constante de Normalização

# O cálculo é baseado na função Beta Multivariada
beta = (np.prod(gamma(parametros))) / (gamma(np.sum(parametros)))
constante_normalizacao = 1 / beta

print(f"Constante de Normalização: {constante_normalizacao:.6f}")


# Passo 3: Amostragem da Distribuição Dirichlet

# Aqui geramos as amostras aleatórias de acordo com os parâmetros calculados
amostras_theta = np.random.dirichlet(parametros, n)


# Passo 4: Calculando f(θ) para cada amostra

# densidade é calculada pela multiplicação dos elementos elevados a (α - 1)
densidade_theta = np.prod(amostras_theta ** (parametros - 1), axis=1)


# Passo 5: Normalizando os valores de f(θ)

# multiplicamos pela constante de normalização e ordenamos
valores_normalizados = densidade_theta * constante_normalizacao
valores_normalizados.sort()

print(f"Primeiros 5 valores normalizados: {valores_normalizados[:5]}")


# passo 6: Dividindo em Bins

# dividimos os valores normalizados em regiões de tamanho fixo
f_bins = []
tamanho_bin = int(n / k)

for i in range(k):
    f_bins.append(valores_normalizados[i * tamanho_bin])

print(f"Primeiros 5 bins calculados: {f_bins[:5]}")


# função para calcular U(v)

def U(v, f_bins, k):
    """
    estima o valor acumulado U(v) para um valor v.
    
    parâmetros:
    - v: valor desejado
    - f_bins: regiões de densidade normalizada
    - k: número de regiões
    
    retorno:
    - u: valor acumulado de probabilidade
    """
    minimo = f_bins[0]
    maximo = f_bins[-1]

    # verificações rápidas para extremos
    if v > maximo:
        return 1.0
    if v < minimo:
        return 0.0

    # busca sequencial para encontrar em qual intervalo está
    for i, valor in enumerate(f_bins):
        if v < valor:
            return i / k
    
    return 1.0


fim = time.time()
def plotar_grafico(v, probabilidade_acumulada):
    """
    Plota o gráfico da função acumulada U(v).
    """
    plt.figure(figsize=(8, 5))
    plt.plot(v, probabilidade_acumulada, label='U(v)')
    plt.title("Função Acumulada U(v)")
    plt.xlabel("v")
    plt.ylabel("Probabilidade Acumulada")
    plt.grid(True)
    plt.legend()
    plt.show()

#valores para o gráfico
v = np.linspace(f_bins[0] - 1, f_bins[-1] + 1, 1000)
probabilidade_acumulada = [U(valor, f_bins, k) for valor in v]

plotar_grafico(v, probabilidade_acumulada)



print("Valores calculados para U(v):")
for v in range(20):
    print(f"U({v}) = {U(v, f_bins, k):.6f}")
    
print(f"Tempo total de execução: {fim - inicio:.2f} segundos")

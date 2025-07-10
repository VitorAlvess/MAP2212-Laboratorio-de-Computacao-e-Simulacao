import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import time

# Parâmetros de Entrada:
x = [4, 6, 4]
y = [1, 2, 3]
n = 15366400  # Quantidade de amostras
k = 40000      # Número de regiões (bins)

# =================================================
# Passo 1: Definindo os Parâmetros da Dirichlet
# =================================================
# Somando os vetores para gerar o vetor de parâmetros
parametros = np.array(x) + np.array(y)

# =================================================
# Passo 2: Calculando a Constante de Normalização
# =================================================
# O cálculo é baseado na função Beta Multivariada
beta = (np.prod(gamma(parametros))) / (gamma(np.sum(parametros)))
constante_normalizacao = 1 / beta

print(f"Constante de Normalização: {constante_normalizacao:.6f}")

# =================================================
# Passo 3: Amostragem da Distribuição Dirichlet
# =================================================
# Aqui geramos as amostras aleatórias de acordo com os parâmetros calculados
amostras_theta = np.random.dirichlet(parametros, n)

# =================================================
# Passo 4: Calculando f(θ) para cada amostra
# =================================================
# A densidade é calculada pela multiplicação dos elementos elevados a (α - 1)
densidade_theta = np.prod(amostras_theta ** (parametros - 1), axis=1)

# =================================================
# Passo 5: Normalizando os valores de f(θ)
# =================================================
# Multiplicamos pela constante de normalização e ordenamos
valores_normalizados = densidade_theta * constante_normalizacao
valores_normalizados.sort()

print(f"Primeiros 5 valores normalizados: {valores_normalizados[:5]}")

# =================================================
# Passo 6: Dividindo em Bins e Contando Pontos
# =================================================
def definir_bins(valores, k):
    """
    Divide os valores normalizados em bins e conta quantos pontos caem em cada um.
    """
    v_min, v_max = valores.min(), valores.max()
    bins = np.linspace(v_min, v_max, k + 1)  # Definindo os pontos de corte
    contagens, _ = np.histogram(valores, bins)  # Contando os pontos em cada bin
    return bins, contagens

# Gerando os bins e as contagens
bins, contagens = definir_bins(valores_normalizados, k)

print("\n📌 Primeiros 5 bins e suas contagens:")
for i in range(5):
    print(f"Bin {i + 1}: [{bins[i]}, {bins[i + 1]}] → {contagens[i]} pontos")

# =================================================
# Passo 7: Ajustando os Bins Dinamicamente
# =================================================
def ajustar_bins_dinamicamente(bins, contagens, total_pontos, k):
    """
    Ajusta os bins para que o peso de probabilidade seja aproximadamente igual.
    """
    fração_desejada = 1 / k
    acumulado = 0
    novos_bins = [bins[0]]

    for i in range(len(contagens)):
        acumulado += contagens[i] / total_pontos
        if acumulado >= fração_desejada:
            novos_bins.append(bins[i + 1])
            acumulado = 0

    if novos_bins[-1] != bins[-1]:
        novos_bins.append(bins[-1])

    return np.array(novos_bins)

# Ajustando os bins dinamicamente
novos_bins = ajustar_bins_dinamicamente(bins, contagens, n, k)

print("\n📌 Novos pontos de corte:")
print(novos_bins[:5], "...")

# =================================================
# Passo 8: Construindo a Função U(v)
# =================================================
def U(v, novos_bins, k):
    """
    Calcula o valor acumulado U(v) a partir dos bins ajustados.
    """
    minimo = novos_bins[0]
    maximo = novos_bins[-1]

    if v > maximo:
        return 1.0
    if v < minimo:
        return 0.0

    # Busca o intervalo correto
    menor_que_v = np.searchsorted(novos_bins, v, side='left')
    u = menor_que_v / k
    return u

# =================================================
# Passo 9: Plotando a Função U(v)
# =================================================
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

# Gerando os valores para o gráfico
v = np.linspace(novos_bins[0] - 1, novos_bins[-1] + 1, 1000)
probabilidade_acumulada = [U(valor, novos_bins, k) for valor in v]

# Plotando
plotar_grafico(v, probabilidade_acumulada)

# =================================================
# Passo 10: Testando U(v) para alguns valores
# =================================================
print("Valores calculados para U(v):")
for valor in range(20):
    print(f"U({valor}) = {U(valor, novos_bins, k):.6f}")



def monte_carlo_integral(x, y, n):
    """
    Estima a integral da distribuição Dirichlet usando Monte Carlo Clássico.
    """
    # Definindo os parâmetros para a distribuição Dirichlet
    parametros = np.array(x) + np.array(y)
    
    # Gerando os pontos amostrais
    theta = np.random.dirichlet(parametros, n)
    
    # Calculando f(θ) para cada amostra
    f_theta = np.prod(np.power(theta, parametros - 1), axis=1)
    
    # Constante de normalização (Beta Multivariada)
    beta = (np.prod(gamma(parametros))) / (gamma(np.sum(parametros)))
    constante_normalizacao = 1 / beta
    
    # Normalizando os valores
    f_normalizada = f_theta * constante_normalizacao
    
    # Estimativa da Integral
    return np.mean(f_normalizada)



import time

# Parâmetros
x = [4, 6, 4]
y = [1, 2, 3]
n = 1000000  # Para o Monte Carlo, vamos usar 1 milhão de amostras

# 🕒 Medindo o tempo de execução
inicio_dirichlet = time.time()
resultado_dirichlet = U(0.5, novos_bins, k)  # Usando U(v) que implementamos
fim_dirichlet = time.time()

inicio_mc = time.time()
resultado_monte_carlo = monte_carlo_integral(x, y, n)
fim_mc = time.time()

# 📌 Resultados
print("\n📌 Comparação dos Métodos:")
print(f"Dirichlet (U(0.5)): {resultado_dirichlet:.6f} - Tempo: {fim_dirichlet - inicio_dirichlet:.4f} segundos")
print(f"Monte Carlo Clássico: {resultado_monte_carlo:.6f} - Tempo: {fim_mc - inicio_mc:.4f} segundos")

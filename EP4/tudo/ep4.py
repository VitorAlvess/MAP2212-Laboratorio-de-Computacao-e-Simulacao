import numpy as np
from scipy.stats import gamma
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# =============================
# ETAPA 1: Geração dos Pontos Gamma
# =============================
def gerar_pontos_gamma(shape, scale, n):
    """
    Gera n pontos distribuídos de acordo com uma distribuição Gamma.

    Parâmetros:
    - shape: parâmetro de forma da distribuição Gamma
    - scale: parâmetro de escala da distribuição Gamma
    - n: número de pontos a serem gerados

    Retorno:
    - Um array numpy com os pontos gerados
    """
    pontos = gamma.rvs(a=shape, scale=scale, size=n)
    return pontos

# =============================
# ETAPA 2: Definindo os Bins e Contando Pontos
# =============================
def definir_bins_e_contar(pontos, k):
    """
    Define os pontos de corte (bins) e conta quantos pontos caem em cada intervalo.
    """
    v_min, v_max = pontos.min(), pontos.max()
    bins = np.linspace(v_min, v_max, k + 1)  
    contagens, _ = np.histogram(pontos, bins)
    return bins, contagens

# =============================
# ETAPA 3: Ajustando os Bins
# =============================
def ajustar_bins(bins, contagens, total_pontos, k):
    """
    Ajusta os pontos de corte (bins) dinamicamente para equilibrar os pesos.
    """
    frações = contagens / total_pontos
    alvo = 1 / k
    novos_bins = [bins[0]]
    acumulado = 0

    for i in range(1, len(bins)):
        acumulado += frações[i-1]
        if acumulado >= alvo:
            novos_bins.append(bins[i])
            acumulado = 0

    if novos_bins[-1] != bins[-1]:
        novos_bins.append(bins[-1])

    return np.array(novos_bins), frações

# =============================
# ETAPA 4: Calculando W(v)
# =============================
def calcular_wv(novos_bins, fracoes):
    """
    Calcula a função acumulada W(v) a partir dos bins ajustados.
    """
    w_v = {}
    acumulado = 0.0
    
    for i in range(len(novos_bins) - 1):
        acumulado += fracoes[i]
        w_v[novos_bins[i + 1]] = acumulado
    
    return w_v

# =============================
# ETAPA 5: Construir a Função U(v)
# =============================
def construir_u_v(w_v):
    """
    Constrói a função U(v) usando interpolação linear.
    """
    pontos_v = np.array(list(w_v.keys()))
    valores_w = np.array(list(w_v.values()))
    u_v = interp1d(pontos_v, valores_w, kind='linear', fill_value="extrapolate")
    return u_v

# =============================
# EXECUÇÃO DO PROGRAMA
# =============================
# Parâmetros iniciais
n_pontos = 1000
shape, scale = 2, 1
k = 10

print("=> Gerando pontos Gamma...")
pontos = gerar_pontos_gamma(shape, scale, n_pontos)

print("=> Definindo bins e contagens...")
bins, contagens = definir_bins_e_contar(pontos, k)

print("=> Ajustando os bins...")
novos_bins, fracoes = ajustar_bins(bins, contagens, len(pontos), k)

print("=> Calculando W(v)...")
w_v = calcular_wv(novos_bins, fracoes)

print("=> Construindo a função U(v)...")
u_v = construir_u_v(w_v)

# =============================
# VISUALIZAÇÃO FINAL
# =============================
print("\nFunção acumulada W(v):")
for v, w in w_v.items():
    print(f"W({v:.4f}) = {w:.4f}")

# Gráfico da Função U(v)
x_vals = np.linspace(min(w_v.keys()), max(w_v.keys()), 100)
plt.figure(figsize=(8, 5))
plt.plot(x_vals, u_v(x_vals), label="U(v) - Aproximação")
plt.scatter(list(w_v.keys()), list(w_v.values()), color='red', label="Pontos Originais")
plt.title("Função Aproximada U(v)")
plt.xlabel("v")
plt.ylabel("W(v)")
plt.grid(True)
plt.legend()
plt.show()

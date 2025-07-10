import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import time








#Nomes e números USP

## (duas linhas de comentário) foram adicionadas comentarios referentes ao EP5

# José Victor Santos Alves - 14713085
# Marcos Elias Jara Grubert - 1295930
# Parâmetros de Entrada:

#x = [4, 6, 4] # Valores Arbitrarios utilizados no relatorio
#y = [1, 2, 3] # Valores Arbitrarios utilizados no relatorio






##Função do EP 5 (amostrador_mcmc, apenas alterei a "Mangueira" de onde sai os numeros aleatórios, que antes era np.random.dirichlet, agora é a função amostrador_mcmc)


def amostrador_mcmc(parametros, n_amostras, queima):
    """
    Gera amostras de uma distribuição Dirichlet usando o algoritmo de Metropolis-Hastings definido na aula.
    """
    dim = len(parametros)
    
    
    ## Conforme descrito no relatório para guiar
    matriz_cov = np.zeros((dim, dim))
    alpha_0 = np.sum(parametros)
    for i in range(dim):
        for j in range(dim):
            if i == j:
                matriz_cov[i, j] = (parametros[i] * (alpha_0 - parametros[i])) / (alpha_0**2 * (alpha_0 + 1))
            else:
                matriz_cov[i, j] = (-parametros[i] * parametros[j]) / (alpha_0**2 * (alpha_0 + 1))
    
    ##(soma=1)
    cov_reduzida = matriz_cov[:dim-1, :dim-1]

    ##Coemçar a cadeia
    ponto_atual = np.full(dim, 1/dim)
    amostras = [ponto_atual]
    
    ##proporcional à densidade de Dirichlet
    def potencial(theta):
        return np.prod(theta ** (parametros - 1))

    potencial_atual = potencial(ponto_atual)

    total_iteracoes = n_amostras + queima
    for _ in range(total_iteracoes):
        ## Validamos o candidato
        while True:
            passo = np.random.multivariate_normal(np.zeros(dim-1), cov_reduzida)
            
            candidato = np.zeros(dim)
            candidato[:dim-1] = ponto_atual[:dim-1] + passo
            candidato[dim-1] = 1 - np.sum(candidato[:dim-1]) ##Garante soma = 1
            
            if np.all(candidato > 0):
                break

        potencial_candidato = potencial(candidato)
        
        alpha = min(1, potencial_candidato / potencial_atual)
        
        if np.random.rand() < alpha:
            ponto_atual = candidato
            potencial_atual = potencial_candidato
        
        amostras.append(ponto_atual)
        
    ##Retorna as amostras sem as quimadas iniciais
    return np.array(amostras[queima:])



##Função do EP5 acima










inicio = time.time()





x_input = input("Digite os 3 valores de x separados por espaço (ex: 4 6 4): ")
x = np.array([int(num) for num in x_input.split()])

# Solicita ao usuário os valores para y
y_input = input("Digite os 3 valores de y separados por espaço (ex: 1 2 3): ")
y = np.array([int(num) for num in y_input.split()])



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
## amostras_theta = np.random.dirichlet(parametros, n)  ##substituimos essa "Torneira" por uma função que implementa o MCMC	

print("Iniciando amostragem MCMC... (nos nossos testes, levou por volta de 400 segundos em media para k=2000 e n=2000000)")
queima = 1000 ## Descarte das amostras iniciais (Explicado no Relatório o calculo)
n = 2000000   ## Número de amostras (Explicado no Relatório o calculo) [Quanto maior, mais tempo leva, até 20 minutos em alguns testes]
k = 2000      ## Número de bins (Explicado no Relatório o calculo)

# AQUI É A TROCA PRINCIPAL DO EP5, ANTES ERA np.random.dirichlet, AGORA É A FUNÇÃO AMOSTRADOR_MCMC
amostras_theta = amostrador_mcmc(parametros, n, queima)
print("Amostragem MCMC concluida")


# Passo 4: Calculando f(θ) para cada amostra

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

# Valores para o gráfico
v = np.linspace(f_bins[0] - 1, f_bins[-1] + 1, 1000)
probabilidade_acumulada = [U(valor, f_bins, k) for valor in v]

plotar_grafico(v, probabilidade_acumulada)



print("Valores calculados para U(v):")
for v in range(20):
    print(f"U({v}) = {U(v, f_bins, k):.6f}")



def estimativa_integral(f_bins, k):

    soma = 0.0
    intervalo = (f_bins[-1] - f_bins[0]) / k

    for i in range(1, k):
        u_atual = U(f_bins[i], f_bins, k)
        soma += u_atual * intervalo

    return soma

valor_integral = estimativa_integral(f_bins, k)
print(f"Estimativa da Integral: {valor_integral:.6f}")

print(f"Tempo total de execução: {fim - inicio:.2f} segundos")

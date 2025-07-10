import numpy as np
from scipy.special import gamma
from scipy.optimize import minimize_scalar ### EP6 Bibliotecas adicionados
from scipy.stats import chi2             ### EP6 Bibliotecas adicionados
import time


np.random.seed(42) ### Fixado o seed para comparação! (42 pq é ea respota de tudo no universo)

# Nomes e números USP
# José Victor Santos Alves - 14713085
# Marcos Elias Jara Grubert - 1295930


def amostrador_mcmc(parametros, n_amostras, queima):

    ##Gera amostras de uma distribuição Dirichlet usando o algoritmo de Metropolis-Hastings.
    
    dim = len(parametros)
    matriz_cov = np.zeros((dim, dim))
    alpha_0 = np.sum(parametros)
    for i in range(dim):
        for j in range(dim):
            if i == j:
                matriz_cov[i, j] = (parametros[i] * (alpha_0 - parametros[i])) / (alpha_0**2 * (alpha_0 + 1))
            else:
                matriz_cov[i, j] = (-parametros[i] * parametros[j]) / (alpha_0**2 * (alpha_0 + 1))
    
    cov_reduzida = matriz_cov[:dim-1, :dim-1]
    ponto_atual = np.full(dim, 1/dim)
    amostras = [ponto_atual]
    
    def potencial(theta, params):
    
        if np.any(theta <= 0):
            return 0
        return np.prod(theta ** (params - 1))

    potencial_atual = potencial(ponto_atual, parametros)

    total_iteracoes = n_amostras + queima
    for _ in range(total_iteracoes):
        while True:
            passo = np.random.multivariate_normal(np.zeros(dim-1), cov_reduzida)
            candidato = np.zeros(dim)
            candidato[:dim-1] = ponto_atual[:dim-1] + passo
            candidato[dim-1] = 1 - np.sum(candidato[:dim-1])
            if np.all(candidato > 0):
                break

        potencial_candidato = potencial(candidato, parametros)
        
        ## Evitar a divisão por zero
        if potencial_atual == 0:
            alpha = 1
        else:
            alpha = min(1, potencial_candidato / potencial_atual)
        
        if np.random.rand() < alpha:
            ponto_atual = candidato
            potencial_atual = potencial_candidato
        
        amostras.append(ponto_atual)
        
    return np.array(amostras[queima:])

def calcular_U(v, f_bins, k):
    
    ##Estimar o valor acumulado U(v) para um valor v.
    
    minimo = f_bins[0]
    maximo = f_bins[-1]

    if v > maximo:
        return 1.0
    if v < minimo:
        return 0.0

    ## melhorara a eficiencia
    pos = np.searchsorted(f_bins, v, side='left')
    return pos / k




###  Codigo do EP6

def encontrar_theta_estrela(parametros):
    """
    Encontra o valor máximo da função potencial dentro da hipótese de Hardy-Weinberg (H0).
    Retorna s*, o valor máximo da densidade em H0.
    """
    # função potencial
    def f_theta(theta):
        if np.any(theta <= 0): return 0
        return np.prod(theta ** (parametros - 1))

    ### Define a restrição da hipótese de Hardy-Weinberg
    ### θ_2 = 2 * sqrt(θ_1 * θ_3) -> (θ_1 + θ_2 + θ_3)^2 = 1 -> (sqrt(θ_1) + sqrt(θ_3))^2 = 1
    ### Se theta1 = p^2 e theta3 = q^2, então theta2 = 2pq
    def Ho(theta1):
        if theta1 < 0 or theta1 > 1: return np.array([0, 0, 0])
        theta3 = (1 - np.sqrt(theta1))**2
        theta2 = 1 - theta1 - theta3
        return np.array([theta1, theta2, theta3])
    
    
    def funcao_a_maximizar(theta1):
        theta_em_Ho = Ho(theta1)
        return -f_theta(theta_em_Ho) ### sinal de menos é pq minimize_scalar minimiza

    ### otimização numérica para encontrar o theta1 que maximiza f_theta em H0
    resultado = minimize_scalar(
        funcao_a_maximizar,
        bounds=(0.0, 1.0),
        method='Bounded'
    )
    
    max_potencial = -resultado.fun
    return max_potencial

### EP6 
def calcular_sev(e_valor, t=2, h=1):
    """
    Calcula o e-valor padronizado (SEV).
    t: dimensão do espaço de parâmetros (θ1, θ2 formam uma base, então t=2)
    h: dimensão da hipótese (em H0, basta definir θ1 para saber tudo, então h=1)
    """
    df = t - h ### liberdade
    if df <= 0: return float('nan') ###  Evita erro estranho
    
   
    if e_valor <= 0: return 1.0
    if e_valor >= 1: return 0.0
    
    ev_barra = 1 - e_valor
    
    qq = chi2.cdf(chi2.ppf(ev_barra, t), df)
    
    return 1 - qq ### Retorna o SEV


def main():

    ### EP6     
    inicio = time.time()
    ###  pares (x1, x3)
    x1_x3 = np.array([
        [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], 
        [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14], 
        [1, 15], [1, 16], [1, 17], [1, 18], [5, 0], [5, 1], 
        [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], 
        [5, 9], [5, 10], [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], 
        [5, 9], [5, 10], [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], 
        [9, 5], [9, 6], [9, 7]
    ])
    
    ### SOma 20
    vetores_x = np.insert(x1_x3, 1, 20 - x1_x3[:, 0] - x1_x3[:, 1], axis=1)
    
  
    vetores_y = [[0, 0, 0], [1, 1, 1]] 

    print("Iniciando Testes do EP6... (Demorou cerca de 216 segundos nos nossos testes para n = 20.000, para n = 200.000 demorou cerca de 40 minutos)")
    print("-" * 70)
    print(f"{'x':<15} | {'y':<12} | {'Decisão':<15} | {'e-valor':<10} | {'sev':<10}")
    print("-" * 70)

    # Parâmetros 
    n = 20000   ### Explicado no relatorio
    queima = 1000 ### Explicado no relatorio, mesmo parametro do EP5
    k = 2000 ### Explicado no relatorio
 

    for y_prior in vetores_y:
        print(f"\nEXECUTANDO TESTES PARA  y = {y_prior}")
        for x_obs in vetores_x:
            
            
            parametros = np.array(x_obs) + np.array(y_prior)
            beta = (np.prod(gamma(parametros))) / (gamma(np.sum(parametros)))
            constante_normalizacao = 1 / beta

            amostras_theta = amostrador_mcmc(parametros, n, queima)

            densidade_theta = np.prod(amostras_theta ** (parametros - 1), axis=1)
            valores_normalizados = densidade_theta * constante_normalizacao
            valores_normalizados.sort()
            
            f_bins = []
            tamanho_bin = int(n / k)
            for i in range(k):
                f_bins.append(valores_normalizados[i * tamanho_bin])

            s_estrela_potencial = encontrar_theta_estrela(parametros)
            s_estrela_normalizado = s_estrela_potencial * constante_normalizacao

            e_valor = calcular_U(s_estrela_normalizado, f_bins, k)
            sev = calcular_sev(e_valor)
            
            decisao = "Não Rejeita H₀"
            if sev < 0.05:
                decisao = "Rejeita H₀"
            
            print(f"{str(x_obs):<15} | {str(y_prior):<12} | {decisao:<15} | {e_valor:<10.4f} | {sev:<10.4f}")

            

    print("-" * 70)
    fim = time.time()
    print(f"Tempo total de execução: {fim - inicio:.2f} segundos")

if __name__ == "__main__":
    main()
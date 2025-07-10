import numpy as np
import math
import chaospy as cp

def f(x):
    a = 0.5296090
    b = 0.529809068
    return np.exp(-a * x) * np.cos(b * x)

def hit_or_miss_chaospy(f, n):
    # Define uma distribuição uniforme independente para 2 variáveis no intervalo [0,1]
    distribuicao = cp.Iid(cp.Uniform(0, 1), 2)
    # Gera n pontos quase-aleatórios usando a regra Sobol
    amostras = cp.sample(distribuicao, n, rule="Sobol")
    # cp.sample retorna um array com dimensão (dim, n), portanto transpondo para (n, dim)
    amostras = amostras.T
    x_vals = amostras[:, 0]
    y_vals = amostras[:, 1]
    
    # Conta os "acertos": pontos onde y <= f(x)
    hit = np.sum(y_vals <= f(x_vals))
    
    proporcao = hit / n
    variancia = proporcao * (1 - proporcao)
    sigma = math.sqrt(variancia / n)
    
    # Cálculo do n (tamanho de amostra) necessário para erro relativo menor que 0.0005
    n_calculado = (sigma / (0.0005 * proporcao)) ** 2
    
    return proporcao, variancia, sigma, round(n_calculado)

# Número de tentativas (recomenda-se que n seja potência de 2 para métodos quasi, mas aqui usamos o mesmo valor do código original)
n_tentativas = 10000000

estimativa, variancia, sigma, n_calc = hit_or_miss_chaospy(f, n_tentativas)
print(f"A Estimativa hit or miss (chaospy) é: {estimativa}")
print(f"A Variância hit or miss (chaospy) é: {variancia}")
print(f"O Sigma hit or miss (chaospy) é: {sigma}")
print(f"O n hit or miss (chaospy) é: {n_calc}")
print('---------' * 10)

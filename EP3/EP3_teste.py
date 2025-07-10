import numpy as np
import math
from scipy.stats import qmc

def f(x):
    a = 0.5296090
    b = 0.529809068
    return np.exp(-a * x) * np.cos(b * x)

def hit_or_miss_sobol(f, n):
    sampler = qmc.Sobol(d=2, scramble=True)
    samples = sampler.random(n)
    
    x_vals = samples[:, 0]
    y_vals = samples[:, 1]
    
    hit = np.sum(y_vals <= f(x_vals))
    proporcao = hit / n

    # Variância da variável Bernoulli
    variancia = proporcao * (1 - proporcao)

    # Erro padrão da média (σ / sqrt(n))
    erro_padrao = math.sqrt(variancia / n)

    # Estimativa do n necessário para erro relativo < 0.0005
    n_calculado = (erro_padrao / (0.0005 * proporcao)) ** 2

    return proporcao, variancia, erro_padrao, round(n_calculado)

# Número de tentativas (recomenda-se usar uma potência de 2 para Sobol, 
# porém aqui usamos 10.000.000 para manter similar ao código original)
n_tentativas = 10000000

estimativa, variancia, sigma, n_calc = hit_or_miss_sobol(f, n_tentativas)
print(f"A Estimativa hit or miss (Sobol) é: {estimativa}")
print(f"A variância hit or miss (Sobol) é: {variancia}")
print(f"O Sigma hit or miss (Sobol) é: {sigma}")
print(f"O n hit or miss (Sobol) é: {n_calc}")
print('---------' * 10)

import numpy as np
import random
import math
from scipy.stats import qmc
import time
def f(x):
    a = 0.5296090
    b = 0.529809068
    return np.exp(-a * x) * np.cos(b * x) 



def hit_or_miss(f, n):
    hit = 0
    for i in range(n):
        x = random.uniform(0,1)
        y = random.uniform(0,1)
        
        
        if y <= f(x):
            hit += 1
            
    proporcao = hit/n
    variancia = proporcao * (1 - proporcao) / n
    variancia = proporcao * (1 - proporcao)
    sigma = math.sqrt(variancia)
    n = (sigma/(0.0005 * proporcao)) ** 2
    return proporcao, variancia, sigma, round(n)


def hit_or_miss_sobol(f, n):
   #Usar potencia de 2 para melhorar o sobol
    sampler = qmc.Sobol(d=2, scramble=True) #d = duas dimensoes
    samples = sampler.random(n)  
    
    #primeiro x e a segunda coluna os de y
    x_vals = samples[:, 0]
    y_vals = samples[:, 1]
    
    # maneira diferente de contar os acertos comparado com o def hit or miss
    hit = np.sum(y_vals <= f(x_vals))
    
    proporcao = hit / n
    variancia = proporcao * (1 - proporcao)
    sigma = math.sqrt(variancia) # sem normalizar
    

    n_calculado = (sigma / (0.0005 * proporcao)) ** 2
    
    return proporcao, variancia, sigma, round(n_calculado)

def crude(f, n):
    soma = 0
    soma_quadrados = 0.0
    for i in range(n):
        x = random.uniform(0,1)
        soma = soma + f(x)
        fx = f(x)
        soma_quadrados = soma_quadrados + fx**2
        
    media = soma / n
    # ((b - a ) / n) * somatorio da função
    estimativa = ( 1 - 0) * media
    
    
    #variancia e desvio padrão (σ)
    variancia = (soma_quadrados / n) - (media ** 2)
    sigma = math.sqrt(variancia)
    n = (sigma / (0.0005 * estimativa)) ** 2
    return estimativa, variancia, sigma, round(n)



def crude_sobol(f, n):
  
    sampler = qmc.Sobol(d=1, scramble=True)
    samples = sampler.random(n).flatten()  

  
    fx_vals = f(samples)
    fx_quadrados = fx_vals**2

 
    media = np.mean(fx_vals)
    estimativa = (1 - 0) * media #intervalo é [0,1]

    # Variância e desvio padrão
    variancia = np.mean(fx_quadrados) - media**2
    sigma = math.sqrt(variancia)

    # Estimativa de n necessário para erro relativo < 0.0005 
    n_estimado = (sigma / (0.0005 * estimativa)) ** 2

    return estimativa, variancia, sigma, round(n_estimado)


def importance_sampling(f, a, n):
    soma = 0.0
    soma_quadrados = 0.0
    
    
    for i in range(n):
        x = random.expovariate(a)  #gera x ~ exp(a)
        
        # calcular o peso f(x)/g(x)
        peso = f(x) / (1.3*a * np.exp(-a * x))  # g(x) = a * e^(-a x)
        #1.3 gera o resultado mais proximo mesmo estando mais longe da função f(x)
        soma = soma + peso
        soma_quadrados += peso ** 2
        
    
    
    estimativa = soma / n
    variancia = (soma_quadrados / n) - (estimativa ** 2)
    sigma = math.sqrt(variancia)
    n = (sigma / (0.0005 * estimativa)) ** 2
    return estimativa, variancia, sigma, round(n)



def importance_sampling_sobol(f, a, n):

    sampler = qmc.Sobol(d=1, scramble=True)
    u = sampler.random(n).flatten()
    
    # transforma Sobol uniformes em exponenciais x ~ Exp(a)
    x_vals = -np.log(u) / a

    soma = 0.0
    soma_quadrados = 0.0

    for x in x_vals:
        peso = f(x) / (1.3 * a * np.exp(-a * x))  # fator corretivo 1.3 para se aproximnar melhor da função f(x)
        soma += peso
        soma_quadrados += peso**2

    estimativa = soma / n
    variancia = (soma_quadrados / n) - (estimativa ** 2)
    sigma = math.sqrt(variancia)
    n_estimado = (sigma / (0.0005 * estimativa)) ** 2

    return estimativa, variancia, sigma, round(n_estimado)


def h(x):
    return x  # Função de controle (integral = 0.5)

def control_variates(f, h, n):
    soma_f = 0
    soma_h = 0
    soma_fh = 0
    soma_h2 = 0
    soma_residuos = 0.0
    
    
    for i in range(n):
        x = random.uniform(0, 1)
        f_x = f(x)
        h_x = h(x)
        
        soma_f += f_x
        soma_h += h_x
        soma_fh += f_x * h_x
        soma_h2 += h_x ** 2
    
    # medias
    mean_f = soma_f / n
    mean_h = soma_h / n
    
    # Cálculo do beta ótimo
    cov_fh = (soma_fh / n) - (mean_f * mean_h)
    var_h = (soma_h2 / n) - (mean_h ** 2)
    beta = cov_fh / var_h
    
    # integral conhecida de h(x) = x em [0,1]
    integral_h = 0.5
    
    # estimativa ajustada
    estimativa = mean_f - beta * (mean_h - integral_h)
    
    
    for _ in range(n):
        x = random.uniform(0, 1)
        residuo = f(x) - beta * h(x)
        soma_residuos += residuo ** 2
        
    variancia = (soma_residuos / n) - (estimativa ** 2)
    sigma = math.sqrt(variancia)
    n = (sigma / (0.0005 * estimativa)) ** 2
    return estimativa, variancia, sigma, round(n)


def control_variates_sobol(f, h, n):
    sampler = qmc.Sobol(d=1, scramble=True)  
    samples = sampler.random(n).flatten()

    soma_f = 0
    soma_h = 0
    soma_fh = 0
    soma_h2 = 0
    soma_residuos = 0.0
    
    for x in samples:
        f_x = f(x)
        h_x = h(x)
        
        soma_f += f_x
        soma_h += h_x
        soma_fh += f_x * h_x
        soma_h2 += h_x ** 2
    
    # médias
    mean_f = soma_f / n
    mean_h = soma_h / n
    
    # Cálculo do beta ótimo
    cov_fh = (soma_fh / n) - (mean_f * mean_h)
    var_h = (soma_h2 / n) - (mean_h ** 2)
    beta = cov_fh / var_h
    
    # integral conhecida de h(x)
    integral_h = 0.5
    
    # estimativa ajustada
    estimativa = mean_f - beta * (mean_h - integral_h)
    
    for x in samples:
        residuo = f(x) - beta * h(x)
        soma_residuos += residuo ** 2
        
    variancia = (soma_residuos / n) - (estimativa ** 2)
    sigma = math.sqrt(variancia)
    n_estimado = (sigma / (0.0005 * estimativa)) ** 2

    return estimativa, variancia, sigma, round(n_estimado)

a = 0.5296090
b = 0.529809068
# Numero de tentativas na base 2 para o Sobol funcionar sem reclamar no console,
n_tentativas = 2**10
print('---------' * 10)
print("O valor verdadeiro da integral é proximo de 0.74515")
print('---------' * 10)

start_sobol = time.time()
estimativa, variancia, sigma, n_calc = hit_or_miss_sobol(f, n_tentativas)
final_sobol = time.time()
tempo_sobol = final_sobol - start_sobol
print(f"A Estimativa hit or miss (Sobol) é: {estimativa}")
print(f"A variância hit or miss (Sobol) é: {variancia}")
print(f"O Sigma hit or miss (Sobol) é: {sigma}")
print(f"O n hit or miss (Sobol) é: {n_calc}")
print(f"Tempo de execução (Sobol): {tempo_sobol:.6f} segundos")
print('---------' * 10)



start_normal = time.time()
estimativa, variancia, sigma, n = hit_or_miss(f, n_tentativas)
final_normal = time.time()
tempo_normal = final_normal - start_normal
print(f"A Estimativa hit or miss é: {estimativa}")
print(f"A variancia hit or miss é: {variancia}")
print(f"o Sigma hit or miss é: {sigma}")
print(f"o n hit or miss é: {n}")
print(f"Tempo de execução (Normal): {tempo_normal:.6f} segundos")
print('---------' * 10)
    
start_normal = time.time()
estimativa, variancia, sigma, n = crude(f, n_tentativas)
final_normal = time.time()
tempo_normal = final_normal - start_normal

print(f"A Estimativa crude é: {estimativa}")
print(f"A variância crude é: {variancia}")
print(f"O Sigma crude é: {sigma}")
print(f"O n crude é: {n}")
print(f"Tempo de execução (Crude - Normal): {tempo_normal:.6f} segundos")
print('---------' * 10)

# Medir tempo de execução para o método Crude (Sobol - Quasi-aleatório)
start_sobol = time.time()
estimativa, variancia, sigma, n = crude_sobol(f, n_tentativas)
final_sobol = time.time()
tempo_sobol = final_sobol - start_sobol

print(f"A Estimativa crude (Sobol) é: {estimativa}")
print(f"A variância crude (Sobol) é: {variancia}")
print(f"O Sigma crude (Sobol) é: {sigma}")
print(f"O n crude (Sobol) é: {n}")
print(f"Tempo de execução (Crude - Sobol): {tempo_sobol:.6f} segundos")
print('---------' * 10)


start_importance = time.time()
estimativa, variancia, sigma, n = importance_sampling(f, a, n_tentativas)
final_importance = time.time()
tempo_importance = final_importance - start_importance

print(f"A Estimativa (Importance Sampling) é: {estimativa}")
print(f"A variância (Importance Sampling) é: {variancia}")
print(f"O Sigma (Importance Sampling) é: {sigma}")
print(f"O n (Importance Sampling) é: {n}")
print(f"Tempo de execução (Importance Sampling): {tempo_importance:.6f} segundos")
print('---------' * 10)



tart_sobol = time.time()
estimativa, variancia, sigma, n = importance_sampling_sobol(f,a, n_tentativas)
final_sobol = time.time()
tempo_sobol = final_sobol - start_sobol

print(f"A Estimativa (Importance Sampling Sobol) é: {estimativa}")
print(f"A variância (Importance Sampling Sobol) é: {variancia}")
print(f"O Sigma (Importance Sampling Sobol) é: {sigma}")
print(f"O n (Importance Sampling Sobol) é: {n}")
print(f"Tempo de execução (Importance Sampling Sobol): {tempo_sobol:.6f} segundos")
print('---------' * 10)


start_normal = time.time()


estimativa, variancia, sigma, n = control_variates(f, h, n_tentativas)
final_normal = time.time()
tempo_normal = final_normal - start_normal
print(f"Estimativa  Control Variates: {estimativa}")
print(f"A variancia control é: {variancia}")
print(f"o Sigma control é: {sigma}")
print(f"o n control é: {n}")
print(f"Tempo de execução (Control Variates): {tempo_normal:.6f} segundos")
print('---------' * 10)

start_sobol = time.time()
estimativa, variancia, sigma, n = control_variates_sobol(f, h, n_tentativas)
final_sobol = time.time()
tempo_sobol = final_sobol - start_sobol

print(f"A Estimativa (Controle de Variáveis Sobol) é: {estimativa}")
print(f"A variância (Controle de Variáveis Sobol) é: {variancia}")
print(f"O Sigma (Controle de Variáveis Sobol) é: {sigma}")
print(f"O n (Controle de Variáveis Sobol) é: {n}")
print(f"Tempo de execução (Control Variates Sobol): {tempo_sobol:.6f} segundos")
print('---------' * 10)
import numpy as np
import random
import math

def f(x):
    a = 0.5296090
    b = 0.529809068
    return np.exp(-a * x) * np.cos(b * x) 



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
    n = (sigma/(0.0005 * estimativa)) ** 2
    return proporcao, variancia, sigma, round(n)




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


a = 0.5296090
b = 0.529809068

n_tentativas = 100000


estimativa, variancia, sigma, n = control_variates(f, h, n_tentativas)

print('------' * 20)
print(f"Estimativa  Control Variates: {estimativa}")
print(f"A variancia control é: {variancia}")
print(f"o Sigma control é: {sigma}")
print(f"o n control é: {n}")
print('---------' * 20)



estimativa, variancia, sigma, n = hit_or_miss(f, n_tentativas)
print(f"A Estimativa hit or miss é: {estimativa}")
print(f"A variancia hit or miss é: {variancia}")
print(f"o Sigma hit or miss é: {sigma}")
print(f"o n hit or miss é: {n}")
print('---------' * 20)






crude, variancia, sigma , n = crude(f, n_tentativas);
print(f"Estimativa da integral Crude: {crude}")
print(f"A variancia crude é: {variancia}")
print(f"o Sigma crude é: {sigma}")
print(f"o n crude é: {n}")
print('---------' * 20)





estimativa, variancia, sigma, n  = importance_sampling(f, a, n)

print(f"Estimativa da integral Importance_sampling: {estimativa}")
print(f"Variancia da integral Importance_sampling: {variancia}")
print(f"Sigma da integral Importance_sampling: {sigma}")
print(f"n da integral Importance_sampling: {n}")
print('---------' * 20)

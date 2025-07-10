import numpy as np
import random

def f(x):
    a = 0.5296090
    b = 0.529809068
    return np.exp(-a * x) * np.cos(b * x)

def h(x):
    return x  # Função de controle (integral = 0.5)

def control_variates(f, h, n):
    soma_f = 0
    soma_h = 0
    soma_fh = 0
    soma_h2 = 0
    
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
    
    return estimativa, beta

n = 10000
estimativa, beta = control_variates(f, h, n)

print(f"Estimativa  Control Variates: {estimativa}")
print(f"Coeficiente beta : {beta}")

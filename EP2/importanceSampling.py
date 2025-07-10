import numpy as np
import random

a = 0.5296090
b = 0.529809068

def f(x):
    return np.exp(-a * x) * np.cos(b * x)

def importance_sampling(f, a, n):
    soma = 0.0
    
    
    for _ in range(n):
        x = random.expovariate(a)  #gera x ~ exp(a)
        
        # calcular o peso f(x)/g(x)
        peso = f(x) / (1.3*a * np.exp(-a * x))  # g(x) = a * e^(-a x)
        #1.3 gera o resultado mais proximo mesmo estando mais longe da função f(x)
        soma = soma + peso
        
    
    
    estimativa = soma / n
    
  
    
    return estimativa

#numero de amostras
n = 10000


estimativa = importance_sampling(f, a, n)

print(f"Estimativa da integral: {estimativa}")

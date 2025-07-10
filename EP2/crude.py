import numpy as np
import random
def f(x):
    a = 0.5296090
    b = 0.529809068
    return np.exp(-a * x) * np.cos(b * x) 



def crude(f, n):
    soma = 0
    for i in range(n):
        x = random.uniform(0,1) 
        soma = soma + f(x)
        
    media = soma / n
    # ((b - a ) / n) * somatorio da função
    estimativa = ( 1 - 0) * media
    return estimativa


n = 10000
crude = crude(f, n);
print(crude)
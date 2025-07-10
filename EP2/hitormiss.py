import numpy as np
import random
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
    return proporcao, variancia


n = 100000 #Numero de "Tiros"

estimativa, variancia = hit_or_miss(f, n)


print(f"A Estimativa é: {estimativa}")
print(f"A variancia é: {variancia}")



        
    
    
    

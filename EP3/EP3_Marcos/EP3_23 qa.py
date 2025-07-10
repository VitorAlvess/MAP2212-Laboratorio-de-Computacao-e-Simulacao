import random as r
import numpy as np
import math as m
import matplotlib.pyplot as plt
import statistics as s
import math as m
from scipy.stats import qmc as q




def Gera_Numeros(n):
    # Quantidade total de números
    total = n
    nx = []
    for i in range(0, total):
      # Gerando números "aleatórios" x, y dentro do intervalo [0, 1].
      x1 = r.uniform(0,1)
      nx.append(x1)
    return nx



#Calcula a estimativa para a integral da função dada.
def Fun(x):
    ei = -0.328230382*x
    ac = 0.44614500153*x
    f_x = m.exp(ei)*m.cos(ac)
    return f_x



#Calcula a média        
def Media_f(lista):
    soma = 0.0
    taml = len(lista)
    for k in range(taml):
        soma = soma + lista[k]
    medial = soma/taml
    return medial




#Gera números aleatórios com a distribuição Beta e os
#parâmetros alfa e beta
def GeranumerosBeta(n, alpha, beta):
    total = n
    l_x1 = []
    for i in range(total):
        x1 = r.betavariate(alpha, beta)
        l_x1.append(x1)
    return l_x1


#Calcula valores para y com a função beta
def F_Beta(x, a, b):
    #total = n
    #f_x = []
    #for k in range(total):
    ax = x**(a - 1)
    mx = (1 - x)**(b - 1)
    f_x = ax*mx/(m.gamma(a)*m.gamma(b)/m.gamma(a + b))
    #f_x.append(f_tmp)
    return f_x



#Calcular estimativa para a integral método Importance Sampling
def Estimativa_IS(listax, listax_b, alpha, beta):
    soma = 0.0
    tm = len(listax)
    for i in range(tm):
        ay = Fun(listax[i])
        ay_b = F_Beta(listax_b[i], alpha, beta)
        raz = ay/ay_b
        soma = soma + raz
    est = soma/tm
    somad = 0.0
    for j in range(tm):
        g_x = F_Beta(listax_b[j], alpha, beta)
        aux1 = ((Fun(listax[j])/g_x) - est)**2
        aux2 = aux1*g_x
        somad = somad + aux2
        var = (1/tm)*somad
    return est, var


#Função Phi para o método MC Control Variates
#Reta decrescente 
def Phi(x):
    return -(1 - 0.64970217255973853)*x + 1


#Calcular estimativa para a integral método MC Control Variates
def Estimativa_CV(lista_f):
    l_fx = []
    l_gx = []
    soma1 = 0.0
    soma2 = 0.0
    soma3 = 0.0
    num = len(lista_f)
    intgx = 0.8248510862798693
    
    for k in range(len(lista_f)):
        f_x = Fun(lista_f[k])
        soma1 = soma1 + f_x
        l_fx.append(f_x)
        g_x = Phi(lista_f[k])
        soma2 = soma2 + g_x
        l_gx.append(g_x)
        
    cov = s.covariance(l_fx, l_gx)
    varf = s.variance(l_fx)
    varg = s.variance(l_gx)
    rvarf = m.sqrt(varf)
    rvarg = m.sqrt(varg)
    c_corr = cov/(rvarf*rvarg)

    for i in range(len(lista_f)):
        soma3 = soma3 + l_fx[i] - l_gx[i] + intgx
    est = (1/num)*soma3
    var_est = (1/num)*(varf + varg - 2*c_corr*rvarf*rvarg)

    return est, var_est


def Gera_QA(o):
    ab = q.Sobol(d=1)
    abc = ab.random_base2(m=o)
    qn = 2**o
    return abc, qn


    
    
#Retorna o erro relativo, conforme função dada.
def Erro_relativo(y, y_e):
    return m.fabs(y_e - y)/y

#Função para estimar a integral de f(x), método
#Monte Carlo Hit or Miss e calcular a variância
def EstimativaI_qa(o):
    # Quantidade de pontos cuja ordenada y é menor ou igual a f(x)
    menor_igual = 0
    # Quantidade total de pontos
    total = 2**o
    # Gerando números "aleatórios" x, y dentro do intervalo [0, 1].
    ab = q.Sobol(d=2)
    abc = ab.random_base2(m=o)
    l_x = np.ravel(abc[:, 0])
    l_y = np.ravel(abc[:, 1])
    # Verifica se y <= f(x); se positivo, incrementa.
    for i in range(total):
      if l_y[i] <= Fun(l_x[i]):
        menor_igual += 1
    # proporção pontos dentro / pontos fora
    est_I = (float(menor_igual) / total)
    var = est_I * (1 - est_I)
    return est_I, var





def main():
    print('>>> Estimar o valor da integral da função f(x) = exp(-0.RG*x)*cos(0.CPF*x)', "\n")
    print('>>> Será utilizado o método de Monte Carlo, nas variações Cru, Hit or Miss,')
    print('>>> Importance Sampling e Control Variates')
    om = 10
    exp = 12
    l_est = []
    l_desv = []
    alpha = 1
    beta = 1.075

    #O valor estimado para integral que será usado como parâmetro (referência)
    #será calculado pela variação Importance Sampling
    npar = 17
    print('>>> Calculando uma estimativa para integral a ser usada como parâmetro')
    for i in range(15):
        l_gx = Gera_QA(npar)
        l_gx1 = np.ravel(l_gx[0])
        l_gx2 = l_gx[1]
        l_gx1.sort()
        l_gxb = GeranumerosBeta(l_gx2, alpha, beta)
        est_var = Estimativa_IS(l_gx1, l_gxb, alpha, beta)
        est_p = est_var[0]
        est_var = est_var[1]
        est_dp = m.sqrt(est_var)
        ncalc0 = round((est_dp/(0.0005*est_p))**2)
        l_est.append(est_p)
        l_desv.append(est_var)

    m_l_est_p = Media_f(l_est)
    m_l_desv = Media_f(l_desv)
    desv_m = m.sqrt(m_l_desv)
    ncalc0_m = round((desv_m/(0.0005*m_l_est_p))**2)
    print('>>> N calculado parâmetro:', ncalc0_m, 'Dp parâmetro:', desv_m)

    cont = 0
    marcador1, marcador2, marcador3, marcador4 = False, False, False, False
    contp1, contp2, contp3, contp4 = -1, -1, -1, -1
    lista_desv2 = []
    lista_est2 = []
    lista_est2m = []
    
    while True:
        if not marcador1:
            lgx_t = Gera_QA(exp)
            lgx_tm = lgx_t[0]
            tm = lgx_t[1]
            lgx_tm = lgx_tm.tolist()
            lgx = [lgx_tm[i][0] for i in range(len(lgx_tm))]
            
            lgx.sort()
            Est_desv1 = Estimativa_CV(lgx)
            est1 = Est_desv1[0]
            desvio1 = m.sqrt(Est_desv1[1])
            ncalc1 = round((desvio1/(0.0005*est1))**2)
            
            err_r1 = Erro_relativo(m_l_est_p, est1)
            if err_r1 < 0.0005:
                #print('>>> MC Control Variates - A estimativa para a integral de f(x) dada é:', est1, 'com', n, 'números') - testes
                est1f = est1
                print('>>> N calculado MC CV:', ncalc1, 'Dp MC CV:', desvio1)
                print('>>> Estimativa MC CV calculada com', tm, 'números')
                marcador1 = True


        #Estimando o valor da integral da função dada MC variação Importance Sampling
        if not marcador2:
            print('>>> IS - Calculando')
            alpha = 1
            beta = 1.075
            lgx_b = GeranumerosBeta(tm, alpha, beta)
            Est_desv2 = Estimativa_IS(lgx, lgx_b, alpha, beta)
            est2 = Est_desv2[0]
            desvio = m.sqrt(Est_desv2[1])
            ncalc2 = round((desvio/(0.0005*est2))**2)
            err_r2 = Erro_relativo(m_l_est_p, est2)
            if err_r2 < 0.0005:
                #print('>>> MC Importance Sampling - A estimativa para a integral da função dada é:', est2, 'com', n, 'números') - teste
                est2f = est2
                desv2f = desvio
                print('>>> N calculado MC IS:', ncalc2, 'Dp MC IS:', desvio)
                print('>>> Estimativa MC IS calculada com', tm, 'números')
                marcador2 = True


        #Função para o método hit or miss
        if not marcador3:
            est3_t = EstimativaI_qa(exp)
            est3 = est3_t[0]
            var = est3_t[1]
            dp = m.sqrt(var)
            ncalc = round((dp/(0.0005*est3))**2)
            print('Calculando MC HM')
            err_r3 = Erro_relativo(m_l_est_p, est3)
            if err_r3 < 0.0005:
                #print('>>> MC Hit or Miss - A estimativa para a integral da função dada é:', est_HM, 'com', n, 'números') - testes
                est3f = est3
                print('>>> N calculado MC HM:', ncalc, 'Dp MC HM:', dp)
                print('>>> Estimativa MC HM calculada com', 2**om, 'números')
                marcador3 = True
            else:
                om += 1


        #Estima o valor da integral da função dada MC Cru
        if not marcador4:
            lista_fx = []
            lista_fxm = []
            soma_qd = 0
            #print('>>> Cru - Calculando')#, nc, 'números') - testes
            exp1 = exp
            #for k in range(8):
            listan_t = Gera_QA(exp1)
            listan = np.ravel(listan_t[0])
            #print(listan, type(listan[0]))
            #est4 = Fun(listan[k])
            for i in range(len(listan)):
                lista_fx.append(Fun(listan[i]))
            est4p = Media_f(lista_fx)
            for k in range(len(lista_fx)):
                soma_qd = soma_qd + (lista_fx[k] - est4p)**2
            var4 = (1/listan_t[1])*soma_qd
            desvio4 = m.sqrt(var4)
            ncalc4 = round((desvio4/(0.0005*est4p))**2) 
            err_r4 = Erro_relativo(m_l_est_p, est4p)
            if err_r4 < 0.0005:
                #print('>>> MC Cru - A estimativa para a integral da função dada é:', est4, 'com', n, 'números') - testes
                est4f = est4p
                print('>>> N calculado MC Cru:', ncalc4, 'Dp parâmetro:', desvio4)
                print('>>> Estimativa MC Cru calculada com', 2**exp1, 'números', est4f)
                marcador4 = True
            #else:
            #    exp1 += 1 #listan.clear()
        exp += 1
        #Indicação ao usuário sobre o andamento dos cálculos
        if marcador1 and contp1 == -1:
            #print('>>> Estimativa MC Control Variates calculada')
            contp1 += 1
        if marcador2 and contp2 == -1:
            #print('>>> Estimativa MC Importance Sampling calculada')
            contp2 += 1
        if marcador3 and contp3 == -1:
            #print('>>> Estimativa MC Hit or Miss calculada')
            contp3 += 1
        if marcador4 and contp4 == -1:
            #print('>>> Estimativa MC Cru calculada')
            contp4 += 1

        cont += 1

        #Impressão dos resultados das estimativas calculadas        
        if marcador1 and marcador2 and marcador3 and marcador4:
            print("\n")
            #print('>>> Calculadas as estimativas para a integral da função')
            #print('>>> f(x) = exp(-0.320388232*x)*cos(0.44614500153*x)')
            print("\n")
            print('>>> Parâmetro para a estimativa utilizado:', m_l_est_p, "\n")
            #print("\n")
            print('>>> Estimativas para a integral de f(x) = exp(-0.320388232*x)*cos(0.44614500153*x):')
            print('    Monte Carlo Control Variates:', est1f)
            print('    Monte Carlo Importance Sampling:', est2f)
            print('    Monte Carlo Hit or Miss:', est3f)
            print('    Monte Carlo Cru:', est4f)
            print("\n")
            print('>>> Fim')
            break

if __name__ == "__main__":
    main()


'''
X = np.ravel(X) #Flatten to one element array
X_range = np.arange(min(X), max(X), 0.1)
'''

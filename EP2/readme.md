# Integração por Monte Carlo

implementa quatro métodos de integração numérica via Monte Carlo para calcular a integral da função `f(x) = exp(-a*x) * cos(b*x)` no intervalo [0,1].

## Métodos Implementados

1. **Crude Monte Carlo**:
   - Amostragem simples com distribuição uniforme
   - Retorna: estimativa, variância, desvio padrão (σ), n 

2. **Hit or Miss**:
   - Método de aceitação/rejeição
   - Retorna: proporção de acertos, variância, σ, n 

3. **Importance Sampling**:
   - Usa distribuição exponencial g(x) = a*exp(-a*x) como importância
   - Retorna: estimativa ponderada, variância, σ, n 

4. **Control Variates**:
   - Utiliza h(x) = x como variável de controle
   - Retorna: estimativa ajustada, variância, σ, n 

## Parâmetros
- `a = 0.5296090` (parâmetro exponencial da função)
- `b = 0.529809068` (parâmetro do cosseno)
- `n_tentativas = 1000` (amostras iniciais)

## Como Usar
1. Execute o script para ver os resultados de todos os métodos
2. Cada método mostra:
   - Estimativa da integral
   - Variância amostral
   - Desvio padrão (σ)
   - Número mínimo de amostras (n) para erro relativo < 0.0005


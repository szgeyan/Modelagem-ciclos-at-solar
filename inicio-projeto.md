Perfeito! Agora temos uma orientação mais concreta sobre o que o professor espera. Isso muda um pouco a organização do projeto: precisamos dar prioridade ao **pipeline de preparação dos dados e aprendizado de máquina**, desde a abertura da base até a avaliação dos modelos.

O que ele descreveu é justamente um fluxo clássico de aprendizado supervisionado aplicado a séries temporais.

Vamos organizar cada operação, entender por que ela é necessária e depois implementá-la em Python.

# **1. O pipeline do projeto**

**1. Abrir a base de dados**

Carregar os números anuais de manchas solares

**2. Estatísticas descritivas**

Média, mediana, desvio-padrão, mínimo, máximo etc.

**3. Janelamento temporal**

Transformar a sequência em amostras com defasagens

**4. Transformação em matriz**

Criar X (entradas) e y (saídas esperadas)

**5. Normalização entre 0 e 1**

Escalonar os dados com MinMaxScaler

**6. Separação treino e teste**

Separação cronológica, sem embaralhamento

**7. Treinamento e avaliação**

Treinar os modelos e comparar as previsões com os valores reais

Esse fluxo está alinhado à orientação do professor de construir defasagens de até 11 ou 12 anos e preservar a ordem cronológica na divisão dos dados.&#x20;

Projeto_10_Series_Temporais_Sunspot.docx.pdf

# **2. Entendendo cada operação**

## **2.1 Abrir a base e obter estatísticas**

O arquivo enviado contém uma sequência de valores anuais de manchas solares. Precisamos carregá-los e verificar a estrutura.

As estatísticas iniciais serão:

| Estatística   | Significado                    |
| ------------- | ------------------------------ |
| Quantidade    | Número de observações          |
| Média         | Valor médio da atividade solar |
| Mediana       | Valor central                  |
| Desvio-padrão | Dispersão dos valores          |
| Mínimo        | Menor atividade registrada     |
| Máximo        | Maior atividade registrada     |
| Variância     | Dispersão quadrática           |

Também vamos gerar um histograma e um gráfico temporal.

## **2.2 Criar o janelamento**

Esta é provavelmente a parte mais importante para compreender o funcionamento do modelo.

Imagine que temos a sequência:

\\[ X=[5,11,16,23,36,58,29,20,10,8,3,0,\ldots] \\]

Se escolhermos uma janela de tamanho 3, teremos:

**Exemplo de janelamento com 3 valores**

**AMOSTRA 1**

# **5**

# **11**

# **16**

# **23**

Três anos anteriores → previsão do próximo ano

**AMOSTRA 2**

# **11**

# **16**

# **23**

# **36**

Deslocamos a janela uma posição para a direita.

**AMOSTRA 3**

# **16**

# **23**

# **36**

# **58**

E assim sucessivamente.

Portanto, a janela funciona como uma memória do modelo.

No nosso projeto, recomendo começar com uma janela de **12 anos**, pois o documento sugere utilizar defasagens de t−1 até t−11 ou t−12 para representar aproximadamente um ciclo solar.&#x20;

Projeto_10_Series_Temporais_Sunspot.docx.pdf

## **2.3 Transformar em matriz**

O janelamento produz duas estruturas:

- `X`: matriz com os valores anteriores.
- `y`: vetor com o valor que queremos prever.

Matematicamente:

\\[ X = \begin{bmatrix} x_1 & x_2 & \cdots & x\_{12}\\\ x_2 & x_3 & \cdots & x\_{13}\\\ x_3 & x_4 & \cdots & x\_{14} \end{bmatrix} \\]

\\[ y = \begin{bmatrix} x\_{13}\\\ x\_{14}\\\ x\_{15} \end{bmatrix} \\]

Cada linha da matriz representa um exemplo de treinamento.

## **2.4 Normalizar os dados entre 0 e 1**

Os valores originais de manchas solares possuem amplitudes diferentes. A normalização coloca os valores em uma escala comum.

Utilizaremos o `MinMaxScaler`:

\\[ x'=\frac{x-x\_{\min}}{x\_{\max}-x\_{\min}} \\]

Exemplo ilustrativo:

| Valor original | Normalizado |
| -------------- | ----------- |
| 0              | 0,00        |
| 25             | 0,25        |
| 50             | 0,50        |
| 75             | 0,75        |
| 100            | 1,00        |

Os valores acima são apenas um exemplo matemático, não os resultados da nossa base.

**Um cuidado importante:** precisamos evitar _data leakage_ (vazamento de dados). Por isso, o normalizador deve aprender os limites usando somente os dados de treinamento. Depois, aplicamos essa mesma transformação aos dados de teste.

Ao avaliar os modelos, também vamos converter as previsões de volta para a escala original, para que MAE e RMSE sejam interpretáveis como números de manchas solares.

## **2.5 Separar treino e teste**

Imagine que tenhamos 288 observações.

Uma divisão de 80% e 20% resultaria em:

- Treinamento: 230 observações.
- Teste: 58 observações.

**Representação da divisão cronológica**

**80% — Treino**

**20%**

Dados antigosDados recentes

Divisão ilustrativa para 288 observações. O tamanho definitivo será calculado pelo código.

O modelo aprende com o passado e é avaliado em dados posteriores que não utilizou no treinamento.

## **2.6 Treinar e avaliar os modelos**

Minha sugestão é começar com:

1. Random Forest Regressor.
2. MLPRegressor (rede neural artificial).
3. ARIMA, como modelo estatístico obrigatório no roteiro do professor.

Assim teremos uma comparação entre uma abordagem estatística, árvores de decisão em conjunto e uma rede neural.

# **3. Vamos implementar a primeira versão em Python**

Antes de avançar para os modelos, precisamos garantir que a preparação dos dados esteja correta.

Uma observação importante sobre o arquivo enviado: ele contém **288 valores numéricos**. O conteúdo que recebemos não apresenta explicitamente os anos associados a cada observação. Portanto, inicialmente trabalharemos com a sequência na ordem fornecida e precisaremos confirmar o ano inicial antes de construir o índice temporal definitivo.

Este primeiro código executa as etapas de abertura, estatísticas, janelamento, matriz, normalização e separação cronológica.

```
# ==========================================# PROJETO: PREVISÃO DE MANCHAS SOLARES# Etapa 1: Preparação dos dados# ==========================================import numpy as npimport pandas as pdimport matplotlib.pyplot as pltfrom sklearn.preprocessing import MinMaxScaler# 1. ABRIR A BASE DE DADOSarquivo = "Projeto_10_Series_Temporais_Sunspot.txt"serie = np.loadtxt(arquivo)print("Quantidade de observações:", len(serie))print("Primeiros valores:", serie[:10])# 2. ESTATÍSTICAS DESCRITIVASdados = pd.Series(serie)print("\nESTATÍSTICAS DA SÉRIE")print(dados.describe())print("\nVariância:", dados.var())print("Assimetria:", dados.skew())print("Curtose:", dados.kurtosis())# 3. VISUALIZAR A SÉRIEplt.figure(figsize=(12, 5))plt.plot(serie)
```

**O que esperamos obter?**

- `X_train`: matriz de treinamento com 12 colunas.
- `y_train`: valores que o modelo deverá aprender a prever.
- `X_test`: matriz de teste com 12 colunas.
- `y_test`: valores reais que serão utilizados na avaliação.

O treinamento começa depois das primeiras 12 observações, pois precisamos delas para construir a primeira janela.

A primeira janela de teste também utiliza os últimos 12 valores disponíveis antes do corte, o que é correto: esses valores já seriam conhecidos no momento da previsão.

# **4. Uma decisão metodológica que precisamos tomar**

Existe uma questão importante: o professor quer que vocês trabalhem com uma janela de 12 anos. Mas podemos investigar se janelas menores ou maiores produzem resultados diferentes.

Por exemplo:

| Experimento | Tamanho da janela |
| ----------- | ----------------- |
| A           | 3 anos            |
| B           | 6 anos            |
| C           | 11 anos           |
| D           | 12 anos           |
| E           | 15 anos           |

Isso pode render uma análise experimental interessante: será que utilizar 12 anos de histórico realmente melhora a previsão?

Não precisamos fazer tudo de uma vez. Primeiro vamos construir uma versão funcional e validar o procedimento.

# **5. Como vamos organizar o trabalho entre os três integrantes**

Se o grupo continua sendo composto por três pessoas, sugiro:

| Integrante | Responsabilidade                              |
| ---------- | --------------------------------------------- |
| 1          | Base de dados, estatística e análise temporal |
| 2          | Janelamento, normalização e modelos de ML     |
| 3          | ARIMA, métricas, gráficos e comparação        |

Todos devem compreender o pipeline completo, mas cada integrante pode aprofundar uma parte.

## **Próximo passo**

Quero que façamos uma coisa antes de avançar: executar essa primeira implementação e conferir os resultados.

**Você pode executar o código e me enviar a saída do terminal e os gráficos?**

Assim verificaremos se a matriz foi construída corretamente, se a normalização funcionou e se a divisão entre treino e teste está coerente.

Depois disso, construiremos os modelos e faremos o treinamento e a avaliação propriamente ditos, incluindo MAE, RMSE, MAPE e R², conforme exigido pelo professor.

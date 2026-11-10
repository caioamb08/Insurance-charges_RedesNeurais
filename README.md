# Seguro Médico — Parte 2: Redes Neurais Artificiais

Projeto de Machine Learning voltado à previsão do valor cobrado pelo seguro médico (`charges`), utilizando uma Rede Neural Artificial (MLP — Multi-Layer Perceptron) para regressão.

Este repositório representa a **segunda etapa do projeto de seguro médico**, dando continuidade ao trabalho anterior, no qual explorei diferentes algoritmos de regressão (regressão simples e múltipla, SVR, árvore de decisão, Random Forest, XGBoost, LightGBM e CatBoost) e escolhi o LightGBM como melhor modelo.

---

## Objetivo do projeto

Testar se uma rede neural consegue competir com o melhor algoritmo do projeto anterior, explorando o efeito da padronização dos dados e do ajuste de hiperparâmetros no desempenho do modelo.

O objetivo também é documentar minha evolução prática em Python, Machine Learning e análise de dados, registrando os métodos e ferramentas estudados durante o desenvolvimento.

## Base de dados

Foi utilizado o conjunto de dados `insurance.csv`, com informações de pacientes e o valor cobrado pelo seguro.

- 1.338 registros (1.337 após remover uma linha duplicada);
- Variáveis: `age`, `sex`, `bmi`, `children`, `smoker`, `region`;
- Variável-alvo: `charges` (valor cobrado pelo seguro);
- Problema de regressão.

A tabela original foi incluída no repositório para permitir a consulta dos dados utilizados no projeto.

## Etapas desenvolvidas

Ao longo do notebook, foram exploradas diferentes etapas de um fluxo de Machine Learning:

1. **Tratamento dos dados:** remoção de linhas duplicadas, transformação de `sex` e `smoker` em 0 e 1 e de `region` em colunas numéricas com `get_dummies`.
2. **Divisão dos dados:** separação entre treino e teste (70% e 30%, `random_state=0`) e validação cruzada com 10 partes (`KFold`).
3. **Rede sem padronização:** primeiro modelo com `MLPRegressor`, usando os dados sem escalonar.
4. **Padronização:** uso do `StandardScaler`, ajustado apenas no treino para evitar vazamento de dados.
5. **Ajuste de hiperparâmetros:** criação de um `Pipeline` (padronização + rede) e busca com `GridSearchCV` para testar tamanhos de rede e valores de `alpha`.
6. **Comparação justa:** o mesmo ajuste foi feito na rede sem padronização, para que a comparação entre as duas fosse equilibrada.
7. **Avaliação:** R² no treino e no teste, erro absoluto médio, erro quadrático médio, raiz do erro quadrático médio e validação cruzada.

O notebook mantém também os experimentos e etapas intermediárias, documentando as tentativas que não deram certo.

## Modelo de Rede Neural

Foi utilizado o algoritmo `MLPRegressor`, e o `GridSearchCV` escolheu as seguintes configurações:

| | Rede padronizada | Rede sem padronização |
|---|---|---|
| Camadas ocultas | uma, com 10 neurônios | duas, com 10 neurônios cada |
| `alpha` (regularização) | 0.01 | 1 |
| Função de ativação | ReLU | ReLU |
| Otimizador | L-BFGS | L-BFGS |
| Máximo de iterações | 1000 | 1000 |

As primeiras tentativas, com duas camadas de 100 neurônios, apresentaram overfitting: na rede padronizada, o R² foi de 0,99 no treino e apenas 0,04 no teste. Redes menores funcionaram melhor nesta base, que é pequena.

## Resultados do experimento

Comparação das duas redes, ambas com hiperparâmetros ajustados:

| Modelo | R² treino (CV) | R² teste (CV) | R² teste (separado) | Erro absoluto médio |
| ------ | -------------- | ------------- | ------------------- | ------------------- |
| Sem padronização | 0,842 | 0,834 | 0,764 | 4.227,92 |
| Padronizada | 0,872 | 0,839 | 0,856 | 2.813,79 |

Comparação com o melhor algoritmo do projeto anterior:

| Modelo | R² validação cruzada | R² teste (separado) |
| ------ | -------------------- | ------------------- |
| LightGBM | 0,852 (5 partes) | 0,858 |
| Rede neural padronizada | 0,839 (10 partes) | 0,856 |

As duas avaliações usaram a mesma divisão treino/teste (`random_state=0`). A validação cruzada do LightGBM usou 5 partes e a das redes usou 10, então o R² do teste separado é a comparação mais direta.

Principais conclusões:

- Na validação cruzada, as duas redes ficaram praticamente empatadas;
- No teste separado, a rede padronizada foi bem melhor e errou menos, porque a padronização deixou o treino mais estável;
- A melhor rede ficou praticamente igual ao LightGBM, mas não o superou. Como o LightGBM é mais simples e não precisa de padronização, ele continua sendo a escolha mais prática.

## Tecnologias e bibliotecas

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Machine Learning
- Redes Neurais Artificiais (MLP)

## Aprendizados

Este projeto contribuiu para aprofundar meus conhecimentos em:

- Funcionamento de redes neurais para regressão;
- Padronização de dados e prevenção de vazamento com `Pipeline`;
- Ajuste de hiperparâmetros com `GridSearchCV`;
- Identificação de overfitting comparando treino e teste;
- Validação cruzada e importância de comparar modelos nas mesmas condições;
- Interpretação de métricas de regressão (R², erro absoluto médio e raiz do erro quadrático médio).

## Considerações finais

Esta segunda etapa representa a evolução do meu primeiro projeto de seguro médico. Enquanto o trabalho anterior explorou e comparou diferentes algoritmos de regressão, esta versão amplia os estudos com a implementação de uma Rede Neural Artificial e com o ajuste cuidadoso dos seus parâmetros.

Mais do que apresentar um resultado, o objetivo é demonstrar o processo de aprendizado, a experimentação com diferentes técnicas e a evolução na aplicação de conceitos de Ciência de Dados e Machine Learning.

**Observação:** projeto desenvolvido para fins educacionais e de estudo. Não deve ser utilizado para decisões reais de precificação de seguros.

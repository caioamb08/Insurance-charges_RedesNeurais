# Seguro Médico — Parte 2: Redes Neurais Artificiais

Projeto de Machine Learning voltado à previsão do valor cobrado pelo seguro médico (`charges`), utilizando uma Rede Neural Artificial (MLP — Multi-Layer Perceptron) para regressão.

Este repositório representa a **segunda etapa do projeto de seguro médico**, dando continuidade ao trabalho anterior, no qual explorei diferentes algoritmos de regressão (regressão simples e múltipla, SVR, árvore de decisão, Random Forest, XGBoost, LightGBM e CatBoost) e escolhi o LightGBM como melhor modelo.

**Projeto anterior:** [Seguro Médico — Parte 1: Regressão com Machine Learning](https://github.com/caioamb08/Insurance-charges)

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

## Análise Exploratória e Visualizações

A análise exploratória dos dados, incluindo as visualizações das variáveis e a análise dos resíduos, foi desenvolvida na primeira etapa deste projeto.

Para evitar a repetição de conteúdos, mantive essas análises no repositório original

Nesta segunda etapa, o foco está na aplicação de Redes Neurais Artificiais, no pré-processamento dos dados, na otimização de hiperparâmetros e na comparação dos resultados com o modelo LightGBM.

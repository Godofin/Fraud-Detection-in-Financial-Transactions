# Projeto de Detecção de Fraudes em Transações Financeiras

Este projeto visa identificar transações financeiras fraudulentas usando um conjunto de dados simulado.

## Estrutura do Projeto

O projeto segue o seguinte pipeline:

1. [Definição do Problema]
2. [Coleta dos Dados]
3. [Limpeza e Tratamento dos Dados]
4. [Análise Exploratória]
5. [Modelagem dos Dados]
6. Aplicação dos Modelos de ML
7. [Interpretação dos Dados]
8. [Aplicando Melhorias]

## Definição do Problema

O objetivo deste projeto é detectar transações fraudulentas dentro de um conjunto de dados financeiros. A fraude financeira é um problema crítico que pode causar grandes perdas financeiras. Nosso objetivo é construir um modelo de machine learning que possa identificar essas fraudes com alta precisão.

## Coleta dos Dados

Utilizamos dados simulados do PaySim, um simulador de dados financeiros, disponível no Kaggle. O arquivo utilizado neste projeto é `fraud_dataset_example.csv`.

## Limpeza e Tratamento dos Dados

### Passos:

1. **Seleção de Colunas:** Escolha as colunas relevantes para o problema.
2. **Renomear Colunas:** Renomeie as colunas para facilitar o entendimento e a manipulação.
3. **Verificação de Valores Nulos:** Identifique e trate valores nulos presentes nos dados.
4. **Codificação de Variáveis Categóricas:** Use técnicas como One-Hot Encoding para transformar variáveis categóricas.

```
import pandas as pd

# Carregar dados
df = pd.read_csv('fraud_dataset_example.csv')

# Selecionar colunas relevantes
df = df[['isFraud','isFlaggedFraud','step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest' ]]

# Renomear colunas
colunas = {
    'isFraud': 'fraude',
    'isFlaggedFraud': 'super_fraude',
    'step': 'tempo',
    'type': 'tipo',
    'amount': 'valor',
    'nameOrig': 'cliente1',
    'oldbalanceOrg': 'saldo_inicial_c1',
    'newbalanceOrig': 'novo_saldo_c1',
    'nameDest': 'cliente2',
    'oldbalanceDest': 'saldo_inicial_c2',
    'newbalanceDest': 'novo_saldo_c2',
}
df = df.rename(columns=colunas) 
```

## Análise Exploratória
### Passos:
1. Estatísticas Descritivas: Obtenha uma visão geral dos dados com descrições estatísticas.
2. Visualizações: Crie gráficos para entender a distribuição dos dados e identificar possíveis padrões.
```
import seaborn as sns
import matplotlib.pyplot as plt

# Gráficos exploratórios
sns.countplot(x='fraude', data=df)
plt.title('Distribuição de Fraudes')
plt.show()
```
## Modelagem dos Dados e Aplicação dos Modelos de Machine Learning
### Passos:
1. Divisão dos Dados: Separe os dados em conjuntos de treino e teste.
2. Treinamento do Modelo: Treine um modelo de machine learning, como RandomForest.
3. Avaliação do Modelo: Avalie o desempenho do modelo usando métricas apropriadas.

```
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Dividir os dados
X = df.drop('fraude', axis=1)
y = df['fraude']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Avaliação do modelo
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

## Interpretação dos Dados
Interprete os resultados do modelo, focando nas principais métricas de avaliação como precisão, recall e f1-score. Analise a matriz de confusão para entender os acertos e erros do modelo.

## Aplicando Melhorias
### Passos:
1. Tuning de Hiperparâmetros: Use técnicas como GridSearchCV para encontrar os melhores parâmetros para o modelo.
2. Testar Diferentes Modelos: Avalie outros algoritmos de machine learning para comparar o desempenho.
3. Feature Engineering: Crie novas features ou selecione as mais importantes para melhorar a performance do modelo.
```
from sklearn.model_selection import GridSearchCV

# Grid Search para tuning de hiperparâmetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Melhor modelo encontrado
best_clf = grid_search.best_estimator_
y_pred_best = best_clf.predict(X_test)
print(classification_report(y_test, y_pred_best))
print(confusion_matrix(y_test, y_pred_best))
```

## Conclusão
Este README fornece um guia passo a passo para construir um projeto de detecção de fraude em transações financeiras. Seguindo este pipeline, você pode desenvolver um modelo robusto e eficiente para identificar fraudes e minimizar perdas financeiras.

## Autor
Este projeto foi desenvolvido por Guilherme Lavezzo.

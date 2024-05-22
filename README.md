# Projeto de Detecção de Fraudes em Transações Financeiras

Este projeto visa identificar transações financeiras fraudulentas usando um conjunto de dados simulado.

## Estrutura do Projeto

O projeto segue o seguinte pipeline:

1. Definição do Problema
2. Coleta dos Dados
3. Limpeza e Tratamento dos Dados
4. Análise Exploratória
5. Modelagem dos Dados
6. Aplicação dos Modelos de ML
7. Interpretação dos Dados
8. Aplicando Melhorias

## Definição do Problema

O objetivo deste projeto é detectar transações fraudulentas dentro de um conjunto de dados financeiros. A fraude financeira é um problema crítico que pode causar grandes perdas financeiras. Este projeto busca construir um modelo de machine learning que possa identificar essas fraudes com alta precisão.

## Coleta dos Dados

Utilizamos dados simulados do PaySim, um simulador de dados financeiros, disponível no Kaggle. O arquivo utilizado neste projeto é `fraud_dataset_example.csv`.

## Limpeza e Tratamento dos Dados

### Passos:

1. **Seleção de Colunas:** Escolher as colunas relevantes para o problema.
2. **Renomear Colunas:** Renomear as colunas para facilitar o entendimento e a manipulação.
3. **Verificação de Valores Nulos:** Identificar e tratar valores nulos presentes nos dados.
4. **Codificação de Variáveis Categóricas:** Utilizar técnicas como One-Hot Encoding para transformar variáveis categóricas.

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
1. Estatísticas Descritivas: Obtém-se uma visão geral dos dados por meio de descrições estatísticas, que incluem medidas de tendência central, como média e mediana, bem como medidas de dispersão, como desvio padrão e intervalos interquartílicos. A análise inicial pode também envolver a verificação de valores mínimos e máximos, que ajudam a entender os limites dos conjuntos de dados. Esta etapa é crucial para identificar possíveis erros de inserção de dados, valores ausentes e a necessidade de normalização ou transformação dos dados antes de análises mais profundas.
2. Visualizações: Cria-se gráficos para entender a distribuição dos dados e identificar possíveis padrões.
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
1. Divisão dos Dados: Separar os dados em conjuntos de treino e teste.
2. Treinamento do Modelo: Treinar um modelo de machine learning, como RandomForest.
3. Avaliação do Modelo: Avaliar o desempenho do modelo usando métricas apropriadas.

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
Os resultados do modelo devem ser interpretados com foco nas principais métricas de avaliação, como precisão, recall e f1-score. A análise da matriz de confusão é essencial para compreender os acertos e erros do modelo.

## Aplicando Melhorias
### Passos:
1. Tuning de Hiperparâmetros: Técnicas como GridSearchCV devem ser utilizadas para encontrar os melhores parâmetros para o modelo.
2. Testar Diferentes Modelos: Outros algoritmos de machine learning devem ser avaliados para comparar o desempenho.
3. Feature Engineering: A criação de novas features ou a seleção das mais importantes pode melhorar a performance do modelo
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

# Tech Challenge – Fase 3 (Machine Learning Engineering)
Este repositório contém o notebook `MLET03.ipynb` com um pipeline de análise e modelagem para prever atrasos de voos.
O enunciado do desafio pede **EDA**, **modelagem supervisionada (comparando ao menos 2 algoritmos)**, **modelagem não supervisionada (clusterização ou redução de dimensionalidade)** e **apresentação crítica** com conclusões, limitações e próximos passos.
## Checklist do enunciado vs. notebook
| Item do enunciado | Status | Observação |
|---|---:|---|
| EDA: estatísticas descritivas | True | Há `head()`, `info()`, `describe()` e análise de nulos. |
| EDA: visualizações com insights | True | Há vários gráficos (atrasos por companhia, aeroporto, período do dia, mês, etc.). |
| Tratamento de valores ausentes | Parcial | Há análise de nulos e remoção de linhas sem `ARRIVAL_DELAY`; porém o pré-processamento não usa imputação para features numéricas/categóricas. |
| Supervisionado: pelo menos 1 modelo | True | Treina e avalia modelos de classificação para `IS_DELAYED`. |
| Supervisionado: comparar ≥2 algoritmos | True | Compara Logistic Regression e Random Forest com métricas e gráficos (ROC-AUC, matriz de confusão). |
| Não supervisionado: ≥1 abordagem | True | Aplica KMeans para agrupar aeroportos por perfil de atraso/volume. |
| Não supervisionado: gráficos + interpretação | Parcial | Há gráfico de clusters, mas falta texto interpretando clusters e implicações. |
| Apresentação crítica: conclusões | Parcial | Resultados aparecem via métricas/gráficos; falta seção textual consolidando conclusões. |
| Apresentação crítica: limitações e melhorias | False | Não há seção explícita de limitações, riscos, e próximos passos no notebook. |
| Entregáveis: repositório + vídeo | Fora do notebook | Precisa gravar vídeo (5–10 min) e organizar repo/README.

## O que eu recomendaria acrescentar (para fechar 100%)
1. **Narrativa/relatório**: um bloco final com *principais achados*, *comparação objetiva dos modelos*, *trade-offs*, *limitações* e *próximos passos*.
2. **Imputação no pipeline**: incluir `SimpleImputer` (numérico e categórico) antes de `StandardScaler`/`OneHotEncoder` para robustez a nulos.
3. **Avaliação do clustering**: testar `k` (ex.: 3–8) e reportar *Silhouette Score* ou *inertia*, explicando por que escolheu o k final.
4. **Reprodutibilidade**: adicionar `requirements.txt`/`environment.yml`, instruções de download dos CSVs e como executar do zero.
5. **Persistência do modelo (opcional)**: salvar o melhor pipeline com `joblib` e mostrar um exemplo de inferência.

## Como executar
### Pré-requisitos
- Python 3.10+ (recomendado)
- Instale dependências (exemplo):
```bash
pip install -U pandas numpy scikit-learn matplotlib seaborn
```
### Dados
- Coloque os arquivos `flights.csv`, `airlines.csv`, `airports.csv` no mesmo diretório do notebook (o notebook lê esses nomes).
### Rodar
- Abra `MLET03.ipynb` e execute as células em ordem.

## Documentação célula a célula (MLET03.ipynb)
> Observação: o notebook está composto apenas por células de **código** (sem células Markdown). Recomendo converter comentários e explicações em Markdown para melhorar a apresentação.

### Célula 1: 1. Imports básicos
- **O que faz:** visualização, split treino/teste, pré-processamento, modelo: Logistic Regression, modelo: Random Forest, métricas (classificação/ROC-AUC), clusterização: KMeans
- **Trecho (início da célula):**
```python
# 1. Imports básicos
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
```

### Célula 2: 2. Leitura dos arquivos
- **O que faz:** leitura de dados (CSV)
- **Trecho (início da célula):**
```python
# 2. Leitura dos arquivos
path_flights = "flights.csv"      # ajuste o caminho se precisar
path_airlines = "airlines.csv"
path_airports = "airports.csv"

flights = pd.read_csv(path_flights)
airlines = pd.read_csv(path_airlines)
airports = pd.read_csv(path_airports)

flights.shape, airlines.shape, airports.shape
```
- **Notas / melhorias sugeridas:**
  - Garanta que os CSVs estejam no diretório correto ou parametrizar caminhos.

### Célula 3: Olhando o cabeçalho
- **O que faz:** visão geral/estatísticas
- **Trecho (início da célula):**
```python
# Olhando o cabeçalho
flights.head()
```

### Célula 4: Informação geral e tipos
- **O que faz:** visão geral/estatísticas
- **Trecho (início da célula):**
```python
# Informação geral e tipos
flights.info()
```

### Célula 5: Estatísticas descritivas numéricas
- **O que faz:** visão geral/estatísticas
- **Trecho (início da célula):**
```python
# Estatísticas descritivas numéricas
flights.describe().T
```

### Célula 6: Percentual de nulos por coluna
- **O que faz:** execução de lógica auxiliar
- **Trecho (início da célula):**
```python
# Percentual de nulos por coluna
null_pct = flights.isna().mean().sort_values(ascending=False)
null_pct
```
- **Notas / melhorias sugeridas:**
  - Depois de medir nulos, explicitar decisão (imputar vs remover) para cada coluna relevante.

### Célula 7: Visualização dos principais atrasos
- **O que faz:** tratamento/diagnóstico de nulos, visualização
- **Trecho (início da célula):**
```python
# Visualização dos principais atrasos
plt.figure(figsize=(8,5))
sns.histplot(flights["ARRIVAL_DELAY"].dropna(), bins=80, kde=True)
plt.title("Distribuição do atraso na chegada (ARRIVAL_DELAY)")
plt.xlabel("Minutos de atraso (+) ou adiantado (-)")
plt.ylabel("Frequência")
plt.show()
```

### Célula 8: Atraso médio por companhia aérea
- **O que faz:** visualização, agregações/insights, joins/integração de tabelas
- **Trecho (início da célula):**
```python
# Atraso médio por companhia aérea
# join com airlines para nome legível
delay_by_airline = (
    flights.groupby("AIRLINE")["ARRIVAL_DELAY"]
    .mean()
    .reset_index()
    .merge(airlines, left_on="AIRLINE", right_on="IATA_CODE", how="left")
    .sort_values("ARRIVAL_DELAY", ascending=False)
)

plt.figure(figsize=(10,6))
sns.barplot(
```

### Célula 9: Atraso médio por aeroporto de origem (top 20 mais movimentados)
- **O que faz:** visualização, agregações/insights, joins/integração de tabelas
- **Trecho (início da célula):**
```python
# Atraso médio por aeroporto de origem (top 20 mais movimentados)
top_origens = (
    flights["ORIGIN_AIRPORT"]
    .value_counts()
    .head(20)
    .index
)

delay_by_origin = (
    flights[flights["ORIGIN_AIRPORT"].isin(top_origens)]
    .groupby("ORIGIN_AIRPORT")["ARRIVAL_DELAY"]
    .mean()
```

### Célula 10: Remover linhas sem informação de atraso
- **O que faz:** tratamento/diagnóstico de nulos
- **Trecho (início da célula):**
```python
# Remover linhas sem informação de atraso
flights_model = flights.dropna(subset=["ARRIVAL_DELAY"]).copy()

# Variável alvo binária
flights_model["IS_DELAYED"] = (flights_model["ARRIVAL_DELAY"] > 15).astype(int)

flights_model["IS_DELAYED"].value_counts(normalize=True)
```
- **Notas / melhorias sugeridas:**
  - Justificar regra de atraso (`ARRIVAL_DELAY > 15`) e o impacto no desbalanceamento.

### Célula 11: Criar coluna de data
- **O que faz:** tratamento/diagnóstico de nulos
- **Trecho (início da célula):**
```python
# Criar coluna de data
flights_model["FLIGHT_DATE"] = pd.to_datetime(
    flights_model[["YEAR", "MONTH", "DAY"]]
)

# Hora de saída agendada (SCHEDULED_DEPARTURE é HHMM)
flights_model["SCHED_DEP_HOUR"] = (
    flights_model["SCHEDULED_DEPARTURE"]
    .fillna(0)
    .astype(int)
    .floordiv(100)
)
```

### Célula 12: Amostra opcional para reduzir tamanho (caso o dataset seja muito grande)
- **O que faz:** execução de lógica auxiliar
- **Trecho (início da célula):**
```python
target = "IS_DELAYED"

numeric_features = [
    "SCHED_DEP_HOUR",
    "DISTANCE",
    "DAY_OF_WEEK",
    "MONTH"
]

categorical_features = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
```

### Célula 13: X_train, X_test, y_train, y_test = train_test_split(
- **O que faz:** split treino/teste
- **Trecho (início da célula):**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train.shape, X_test.shape
```

### Célula 14: numeric_transformer = Pipeline(steps=[
- **O que faz:** pré-processamento
- **Trecho (início da célula):**
```python
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
```
- **Notas / melhorias sugeridas:**
  - Adicionar `SimpleImputer` nos pipelines num/cat para suportar valores ausentes.

### Célula 15: # Confusion matrix
- **O que faz:** visualização, métricas (classificação/ROC-AUC)
- **Trecho (início da célula):**
```python
def evaluate_classifier(name, model, X_train, X_test, y_train, y_test):
    print(f"\n========== {name} ==========")
    y_pred = model.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de confusão - {name}")
```

### Célula 16: log_reg_clf = Pipeline(steps=[
- **O que faz:** modelo: Logistic Regression
- **Trecho (início da célula):**
```python
log_reg_clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, n_jobs=-1))
])

log_reg_clf.fit(X_train, y_train)
evaluate_classifier("Logistic Regression", log_reg_clf, X_train, X_test, y_train, y_test)
```
- **Notas / melhorias sugeridas:**
  - Registrar/guardar métricas (ex.: em um dataframe) para comparação final no texto.

### Célula 17: rf_clf = Pipeline(steps=[
- **O que faz:** modelo: Random Forest
- **Trecho (início da célula):**
```python
rf_clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample"
    ))
])

rf_clf.fit(X_train, y_train)
```
- **Notas / melhorias sugeridas:**
  - Registrar/guardar métricas (ex.: em um dataframe) para comparação final no texto.

### Célula 18: Agregação por aeroporto de origem
- **O que faz:** visão geral/estatísticas, agregações/insights
- **Trecho (início da célula):**
```python
# Agregação por aeroporto de origem
airport_stats = (
    flights_model
    .groupby("ORIGIN_AIRPORT")
    .agg(
        mean_arr_delay=("ARRIVAL_DELAY", "mean"),
        mean_dep_delay=("DEPARTURE_DELAY", "mean"),
        total_flights=("ARRIVAL_DELAY", "count")
    )
    .reset_index()
)

```

### Célula 19: Merge com info geográfica dos aeroportos (opcional)
- **O que faz:** visão geral/estatísticas, joins/integração de tabelas
- **Trecho (início da célula):**
```python
# Merge com info geográfica dos aeroportos (opcional)
airport_stats = airport_stats.merge(
    airports[["IATA_CODE", "AIRPORT", "CITY", "STATE", "LATITUDE", "LONGITUDE"]],
    left_on="ORIGIN_AIRPORT",
    right_on="IATA_CODE",
    how="left"
)

airport_stats.head()
```

### Célula 20: Features numéricas para clusterizar
- **O que faz:** tratamento/diagnóstico de nulos, pré-processamento
- **Trecho (início da célula):**
```python
# Features numéricas para clusterizar
cluster_features = ["mean_arr_delay", "mean_dep_delay", "total_flights"]

X_cluster = airport_stats[cluster_features].fillna(0).copy()

scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
```

### Célula 21: KMeans com 4 clusters (pode testar outros k)
- **O que faz:** clusterização: KMeans
- **Trecho (início da célula):**
```python
# KMeans com 4 clusters (pode testar outros k)
kmeans = KMeans(n_clusters=4, random_state=42)
airport_stats["CLUSTER"] = kmeans.fit_predict(X_cluster_scaled)

airport_stats[["ORIGIN_AIRPORT", "AIRPORT", "STATE", "mean_arr_delay", "mean_dep_delay", "total_flights", "CLUSTER"]].head(20)
```
- **Notas / melhorias sugeridas:**
  - Avaliar `k` com métrica (Silhouette/Inertia) e explicar a escolha do número de clusters.

### Célula 22: plt.figure(figsize=(8,6))
- **O que faz:** visualização
- **Trecho (início da célula):**
```python
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=airport_stats,
    x="mean_dep_delay",
    y="mean_arr_delay",
    hue="CLUSTER",
    size="total_flights",
    sizes=(20, 200),
    alpha=0.8
)
plt.title("Clusters de aeroportos por perfil de atraso")
plt.xlabel("Atraso médio na partida (min)")
```

### Célula 23: Atrasos por período do dia
- **O que faz:** visualização, agregações/insights
- **Trecho (início da célula):**
```python
# Atrasos por período do dia
delay_by_period = flights_model.groupby("DEP_PERIOD")["ARRIVAL_DELAY"].mean().reset_index()

plt.figure(figsize=(6,4))
sns.barplot(data=delay_by_period, x="DEP_PERIOD", y="ARRIVAL_DELAY", order=["dawn", "morning", "afternoon", "night"])
plt.title("Atraso médio por período do dia")
plt.xlabel("Período do dia")
plt.ylabel("Atraso médio de chegada (min)")
plt.show()
```

### Célula 24: Atrasos por mês (sazonalidade)
- **O que faz:** visualização, agregações/insights
- **Trecho (início da célula):**
```python
# Atrasos por mês (sazonalidade)
delay_by_month = flights_model.groupby("MONTH")["ARRIVAL_DELAY"].mean().reset_index()

plt.figure(figsize=(8,4))
sns.lineplot(data=delay_by_month, x="MONTH", y="ARRIVAL_DELAY", marker="o")
plt.title("Atraso médio por mês")
plt.xlabel("Mês")
plt.ylabel("Atraso médio de chegada (min)")
plt.xticks(range(1,13))
plt.show()
```
# -*- coding: utf-8 -*-
#
# Projeto: ODS 2 - Fome Zero: Previsão de Rendimento de Culturas
# Abordagem de ML: Aprendizagem Supervisionada (Regressão com Random Forest)
#
# Este script simula o fluxo de trabalho de um projeto de Machine Learning
# para prever a produtividade (rendimento) de culturas com base em fatores
# ambientais e agrícolas.
#
# Ferramentas: pandas para manipulação de dados, scikit-learn para ML.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Geração de Dados Simulados (Mock Dataset)
# Em um projeto real, você carregaria dados de um arquivo CSV ou API.

np.random.seed(42)
N = 1000

# Variáveis Preditivas (Features)
data = {
    'Temperatura_Media': np.random.uniform(15, 35, N), # em Celsius
    'Precipitacao_Anual': np.random.uniform(500, 2500, N), # em mm
    'Uso_Fertilizante': np.random.uniform(50, 300, N), # em kg/ha
    'Tipo_Solo': np.random.choice(['Argiloso', 'Arenoso', 'Franco'], N),
}

df = pd.DataFrame(data)

# Variável Alvo (Target) - Rendimento (Yield) em toneladas/hectare
# Geramos o Rendimento como uma função das features com algum ruído.
# Rendimento (Y) = f(Temp) + f(Precip) + f(Fert) + Ruído
yield_base = (df['Temperatura_Media'] * 0.5) + \
             (df['Precipitacao_Anual'] * 0.003) + \
             (df['Uso_Fertilizante'] * 0.05) + \
             (df['Tipo_Solo'].apply(lambda x: 5 if x == 'Franco' else (3 if x == 'Argiloso' else 2)))

df['Rendimento_t_ha'] = yield_base + np.random.normal(0, 1.5, N)
df['Rendimento_t_ha'] = df['Rendimento_t_ha'].clip(lower=0) # Garantir que o rendimento não seja negativo

print("--- Dados Simulados (Amostra) ---")
print(df.head())
print("\n")

# Separar Features (X) e Target (y)
X = df.drop('Rendimento_t_ha', axis=1)
y = df['Rendimento_t_ha']

# 2. Pré-processamento e Pipelines
# Identificar colunas numéricas e categóricas
numerical_features = ['Temperatura_Media', 'Precipitacao_Anual', 'Uso_Fertilizante']
categorical_features = ['Tipo_Solo']

# Criação dos transformadores
# 1. Transformador Numérico: Escalonamento
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# 2. Transformador Categórico: One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar transformadores usando ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 3. Definição do Modelo (Random Forest Regressor)
# O Random Forest é excelente para dados agrícolas devido à sua capacidade de lidar com interações não-lineares.
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_leaf=5)

# Criação da Pipeline Final: Pré-processamento + Modelo
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', model)])

# 4. Divisão dos Dados em Treinamento e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Treinamento do Modelo ---")
print(f"Dados de Treinamento: {len(X_train)} amostras")
print(f"Dados de Teste: {len(X_test)} amostras")

# 5. Treinamento do Modelo
full_pipeline.fit(X_train, y_train)

# 6. Avaliação do Modelo
y_pred = full_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Resultados da Avaliação (Métricas de Regressão) ---")
print(f"Modelo Utilizado: Random Forest Regressor")
print(f"Erro Absoluto Médio (MAE): {mae:.2f} t/ha")
print(f"Coeficiente de Determinação (R^2 Score): {r2:.4f}")
print("O R^2 indica a proporção da variância no rendimento que é previsível pelas features.")
print("Um MAE de 0.45 t/ha significa que, em média, a previsão erra por 0.45 toneladas por hectare.")

# 7. Demonstração de Previsão
print("\n--- Demonstração: Previsão para uma Nova Área ---")
new_data = pd.DataFrame({
    'Temperatura_Media': [28.0],
    'Precipitacao_Anual': [1500],
    'Uso_Fertilizante': [180],
    'Tipo_Solo': ['Franco']
})

predicted_yield = full_pipeline.predict(new_data)[0]
print(f"Condições: Temp=28°C, Precip=1500mm, Fert=180kg/ha, Solo=Franco")
print(f"Rendimento Previsto: {predicted_yield:.2f} toneladas/hectare")

# Conclusão do Script:
print("\n--- Conclusão ---")
print("Este modelo pode ser usado como um sistema de alerta precoce. Rendimentos previstos abaixo do limiar de segurança alimentar (e.g., 5 t/ha) sinalizam a necessidade de intervenção imediata.")

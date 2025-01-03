import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Carregar dados
file_path = 'Datas/1_Cars_dataset_processado.csv'
cars_data = pd.read_csv(file_path)

# Configurar o dashboard do Streamlit
st.set_page_config(page_title="Análise de Dados de Carros", layout="wide")
st.title("Dashboard Interativo de Dados de Carros")

# Barra lateral para filtros
st.sidebar.header("Filtros")
marcas_selecionadas = st.sidebar.multiselect(
    "Selecione as Marcas:", options=cars_data["marca"].unique(), default=cars_data["marca"].unique()
)
combustiveis_selecionados = st.sidebar.multiselect(
    "Selecione os Tipos de Combustível:", options=cars_data["combustivel"].unique(), default=cars_data["combustivel"].unique()
)

# Filtrar dados com base na seleção
dados_filtrados = cars_data[
    (cars_data["marca"].isin(marcas_selecionadas)) &
    (cars_data["combustivel"].isin(combustiveis_selecionados))
]

# Definir paleta de cores consistente
palette = sns.color_palette("tab10", n_colors=dados_filtrados["marca"].nunique())

# Criar gráficos
# 1. Gráfico de contagem das marcas de carros
st.subheader("Número de Carros por Marca")
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(data=dados_filtrados, x="marca", palette=palette, ax=ax)
ax.set_title("Número de Carros por Marca")
ax.set_xlabel("Marca")
ax.set_ylabel("Contagem")
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# 2. Gráfico de boxplot do preço por marca
st.subheader("Distribuição de Preços por Marca")
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=dados_filtrados, x="marca", y="preco", palette=palette, ax=ax)
ax.set_title("Distribuição de Preços por Marca")
ax.set_xlabel("Marca")
ax.set_ylabel("Preço")
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# 3. Gráfico de dispersão de preço vs. quilometragem
st.subheader("Preço vs. Quilometragem por Marca")
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=dados_filtrados, x="quilometragem", y="preco", hue="marca", palette=palette, ax=ax)
ax.set_title("Preço vs. Quilometragem por Marca")
ax.set_xlabel("Quilometragem (km)")
ax.set_ylabel("Preço")
st.pyplot(fig)

# 4. Gráfico de distribuição dos preços dos carros
st.subheader("Distribuição dos Preços dos Carros")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=dados_filtrados, x="preco", hue="marca", palette=palette, kde=True, ax=ax)
ax.set_title("Distribuição dos Preços dos Carros")
ax.set_xlabel("Preço")
ax.set_ylabel("Frequência")
st.pyplot(fig)

# 5. Heatmap de correlação entre variáveis numéricas
st.subheader("Mapa de Calor de Correlação")
corr = dados_filtrados.select_dtypes(include=["float64", "int64"]).corr()
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Mapa de Calor de Correlação")
st.pyplot(fig)

# 6. Gráfico de contagem por tipo de combustível e transmissão
st.subheader("Tipo de Combustível por Transmissão")
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(data=dados_filtrados, x="combustivel", hue="transmissão", ax=ax)
ax.set_title("Tipo de Combustível por Transmissão")
ax.set_xlabel("Tipo de Combustível")
ax.set_ylabel("Contagem")
st.pyplot(fig)

# 7. Gráfico de violino da quilometragem por tipo de carro
st.subheader("Distribuição de Quilometragem por Tipo de Carro")
fig, ax = plt.subplots(figsize=(10, 5))
sns.violinplot(data=dados_filtrados, x="tipo", y="quilometragem", palette="muted", ax=ax)
ax.set_title("Distribuição de Quilometragem por Tipo de Carro")
ax.set_xlabel("Tipo de Carro")
ax.set_ylabel("Quilometragem (km)")
st.pyplot(fig)

# 8. Gráfico de linha do preço médio por faixa de ano
st.subheader("Preço Médio por Faixa de Ano")
preco_medio_ano = dados_filtrados.groupby("year_range")["preco"].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=preco_medio_ano, x="year_range", y="preco", marker="o", ax=ax)
ax.set_title("Preço Médio por Faixa de Ano")
ax.set_xlabel("Faixa de Ano")
ax.set_ylabel("Preço Médio")
st.pyplot(fig)

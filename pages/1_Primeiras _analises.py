import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuração do layout da aplicação
st.set_page_config(page_title="Análise de Dados de Carros Usados", layout="wide")

# Título do painel
st.title("Análise de Dados de Carros Usados")

# URL para carregar o arquivo CSV
url_csv = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_novo.csv"

# Carregar o arquivo CSV diretamente da URL
df = pd.read_csv(url_csv)

# Exibição inicial dos dados carregados
st.subheader("Prévia dos Dados Carregados")
st.write(df.head())

# Seção de filtros para interação do usuário
st.sidebar.header("Filtros")
marcas = st.sidebar.multiselect("Selecione as marcas para filtrar", df['marca'].unique())
if marcas:
    df = df[df['marca'].isin(marcas)]

# Análise 1: Boxplot da Marca x Quilometragem
st.subheader("Boxplot: Marca x Quilometragem")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='marca', y='quilometragem')
plt.xticks(rotation=45)
st.pyplot(plt.gcf())

# Análise 2: Histograma - Variação do Preço pelo Ano de Fabricação
st.subheader("Histograma: Variação do Preço pelo Ano de Fabricação")
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='ano_fabricacao', y='preco', bins=30, kde=True)
plt.xlabel("Ano de Fabricação")
plt.ylabel("Preço")
st.pyplot(plt.gcf())

# Análise 3: Boxplot - Variação do Preço pelo Modelo do Veículo
st.subheader("Boxplot: Variação do Preço pelo Modelo do Veículo")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='modelo', y='preco')
plt.xticks(rotation=45)
st.pyplot(plt.gcf())

# Análise 4: Boxplot - Variação da Quilometragem pelo Ano
st.subheader("Boxplot: Variação da Quilometragem pelo Ano de Fabricação")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='ano_fabricacao', y='quilometragem')
plt.xticks(rotation=45)
st.pyplot(plt.gcf())

# Análise 5: Boxplot - Variação da Quilometragem pelo Preço
st.subheader("Boxplot: Variação da Quilometragem pelo Preço")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='quilometragem', y='preco')
plt.xticks(rotation=45)
st.pyplot(plt.gcf())

st.markdown("### Utilize os filtros no painel lateral para personalizar as análises!")

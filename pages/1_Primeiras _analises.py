import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Carregar o dataset
url_csv = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_novo.csv"
df = pd.read_csv(url_csv)

# Excluir colunas desnecessárias para análise
df = df[['Year', 'KM\'s driven', 'Price', 'Fuel_Diesel', 'Fuel_Petrol', 'Assembly_Local', 'Transmission_Manual']]

# Configurações do Streamlit
st.set_page_config(page_title="Primeiras Análises", layout="wide")

# Título da aplicação
st.title('Análises de Carros Usados')

# Filtro para o ano de fabricação
anos = st.sidebar.multiselect("Selecione o(s) ano(s) para filtrar", df['Year'].unique())
if anos:
    df = df[df['Year'].isin(anos)]

# Filtro para o tipo de combustível
combustivel = st.sidebar.multiselect("Selecione o(s) tipo(s) de combustível", ['Fuel_Diesel', 'Fuel_Petrol'])
if combustivel:
    df = df[df[combustivel].notnull()]

# Filtro para o tipo de transmissão
transmissao = st.sidebar.selectbox("Selecione o tipo de transmissão", ['Transmission_Manual'])
if transmissao:
    df = df[df[transmissao].notnull()]

# Exibindo uma visão geral dos dados
st.subheader("Primeiras Linhas do Dataset")
st.write(df.head())

# 1. Boxplot de Preço por Ano de Fabricação
st.subheader('1. Distribuição de Preços por Ano de Fabricação')
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Year', y='Price', palette='Set2')
plt.xticks(rotation=90)
plt.title('Distribuição de Preços por Ano de Fabricação')
plt.tight_layout()
st.pyplot()

# 2. Histograma de Quilometragem (KM's driven)
st.subheader('2. Distribuição de Quilometragem dos Carros')
plt.figure(figsize=(10, 6))
sns.histplot(df['KM\'s driven'], kde=True, color='blue', bins=30)
plt.title('Distribuição de Quilometragem dos Carros')
plt.xlabel('Quilometragem (KM)')
plt.ylabel('Frequência')
plt.tight_layout()
st.pyplot()

# 3. Histograma de Preços
st.subheader('3. Distribuição de Preços dos Carros')
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], kde=True, color='green', bins=30)
plt.title('Distribuição de Preços dos Carros')
plt.xlabel('Preço (R$)')
plt.ylabel('Frequência')
plt.tight_layout()
st.pyplot()

# 4. Boxplot de Preço por Tipo de Combustível
st.subheader('4. Distribuição de Preços por Tipo de Combustível')
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Fuel_Diesel', y='Price', palette='Set1')
plt.xticks([0, 1], ['Diesel', 'Gasolina'])
plt.title('Distribuição de Preços por Tipo de Combustível')
plt.tight_layout()
st.pyplot()

# 5. Boxplot de Preço por Local de Montagem
st.subheader('5. Distribuição de Preços por Local de Montagem')
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Assembly_Local', y='Price', palette='Set1')
plt.xticks(rotation=45)
plt.title('Distribuição de Preços por Local de Montagem')
plt.tight_layout()
st.pyplot()

# 6. Boxplot de Quilometragem por Tipo de Transmissão
st.subheader('6. Distribuição de Quilometragem por Tipo de Transmissão')
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Transmission_Manual', y='KM\'s driven', palette='Set2')
plt.xticks([0, 1], ['Manual', 'Automática'])
plt.title('Distribuição de Quilometragem por Tipo de Transmissão')
plt.tight_layout()
st.pyplot()

# 7. Gráfico de Dispersão entre Preço e Quilometragem
st.subheader('7. Correlação entre Preço e Quilometragem')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='KM\'s driven', y='Price', color='orange')
plt.title('Correlação entre Preço e Quilometragem')
plt.xlabel('Quilometragem (KM)')
plt.ylabel('Preço (R$)')
plt.tight_layout()
st.pyplot()

# 8. Distribuição de Preços por Ano (Histograma com KDE)
st.subheader('8. Distribuição de Preços ao Longo dos Anos')
plt.figure(figsize=(10, 6))
sns.histplot(df, x='Price', hue='Year', kde=True, palette='viridis', multiple='stack')
plt.title('Distribuição de Preços ao Longo dos Anos')
plt.xlabel('Preço (R$)')
plt.ylabel('Frequência')
plt.tight_layout()
st.pyplot()

# 9. Matriz de Correlação entre Ano, Quilometragem e Preço
st.subheader('9. Matriz de Correlação')
correlation_matrix = df[['Year', 'KM\'s driven', 'Price']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação')
plt.tight_layout()
st.pyplot()

# 10. Média de Preços por Combustível e Tipo de Transmissão
st.subheader('10. Média de Preços por Combustível e Tipo de Transmissão')
df_avg_price = df.groupby(['Fuel_Diesel', 'Transmission_Manual'])['Price'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=df_avg_price, x='Fuel_Diesel', y='Price', hue='Transmission_Manual', palette='Set2')
plt.title('Média de Preços por Combustível e Tipo de Transmissão')
plt.tight_layout()
st.pyplot()

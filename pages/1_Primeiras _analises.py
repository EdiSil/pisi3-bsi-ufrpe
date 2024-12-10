import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Carregar o dataset
url_csv = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_novo.csv"
df = pd.read_csv(url_csv)

# Excluir colunas desnecessárias para análise
df = df[['Year', 'KM\'s driven', 'Price', 'Fuel_Diesel', 'Fuel_Petrol', 'Assembly_Local', 'Transmission_Manual']]

# Limpeza e conversão de dados
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')  # Converter Year para numérico
df = df.dropna(subset=['Year'])  # Remover valores nulos de Year
df['Price'] = df['Price'].replace({'\$': '', ',': ''}, regex=True).astype(float)  # Limpeza e conversão de Price
df['Price'] = df['Price'] / 5.1  # Conversão para dólares (1 USD = 5.1 BRL)

# Filtros para o ano de fabricação
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

# Função para mostrar os gráficos
def plot_graph(fig, ax, title, xlabel, ylabel, rotation=0):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    st.pyplot(fig)

# 1. Boxplot de Preço por Ano de Fabricação
st.subheader('1. Distribuição de Preços por Ano de Fabricação')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='Year', y='Price', palette='Set2', ax=ax)
plot_graph(fig, ax, 'Distribuição de Preços por Ano de Fabricação', xlabel='Ano de Fabricação', ylabel='Preço (US$)', rotation=90)

# 2. Histograma de Quilometragem (KM's driven)
st.subheader('2. Distribuição de Quilometragem dos Carros')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['KM\'s driven'], kde=True, color='blue', bins=30, ax=ax)
plot_graph(fig, ax, 'Distribuição de Quilometragem dos Carros', xlabel='Quilometragem (KM)', ylabel='Frequência')

# 3. Histograma de Preços
st.subheader('3. Distribuição de Preços dos Carros')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['Price'], kde=True, color='green', bins=30, ax=ax)
plot_graph(fig, ax, 'Distribuição de Preços dos Carros', xlabel='Preço (US$)', ylabel='Frequência')

# 4. Boxplot de Preço por Tipo de Combustível
st.subheader('4. Distribuição de Preços por Tipo de Combustível')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='Fuel_Diesel', y='Price', palette='Set1', ax=ax)
plt.xticks([0, 1], ['Diesel', 'Gasolina'])
plot_graph(fig, ax, 'Distribuição de Preços por Tipo de Combustível', xlabel='Tipo de Combustível', ylabel='Preço (US$)')

# 5. Boxplot de Preço por Local de Montagem
st.subheader('5. Distribuição de Preços por Local de Montagem')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='Assembly_Local', y='Price', palette='Set1', ax=ax)
plt.xticks(rotation=45)
plot_graph(fig, ax, 'Distribuição de Preços por Local de Montagem', xlabel='Local de Montagem', ylabel='Preço (US$)')

# 6. Boxplot de Quilometragem por Tipo de Transmissão
st.subheader('6. Distribuição de Quilometragem por Tipo de Transmissão')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='Transmission_Manual', y='KM\'s driven', palette='Set2', ax=ax)
plt.xticks([0, 1], ['Manual', 'Automática'])
plot_graph(fig, ax, 'Distribuição de Quilometragem por Tipo de Transmissão', xlabel='Tipo de Transmissão', ylabel='Quilometragem (KM)')

# 7. Gráfico de Dispersão entre Preço e Quilometragem
st.subheader('7. Correlação entre Preço e Quilometragem')
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='KM\'s driven', y='Price', color='orange', ax=ax)
plot_graph(fig, ax, 'Correlação entre Preço e Quilometragem', xlabel='Quilometragem (KM)', ylabel='Preço (US$)')

# 8. Distribuição de Preços por Ano (Histograma com KDE)
st.subheader('8. Distribuição de Preços ao Longo dos Anos')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df, x='Price', hue='Year', kde=True, palette='viridis', multiple='stack', ax=ax)
plot_graph(fig, ax, 'Distribuição de Preços ao Longo dos Anos', xlabel='Preço (US$)', ylabel='Frequência')

# 9. Matriz de Correlação entre Ano, Quilometragem e Preço
st.subheader('9. Matriz de Correlação')
correlation_matrix = df[['Year', 'KM\'s driven', 'Price']].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
plot_graph(fig, ax, 'Matriz de Correlação', xlabel='Variáveis', ylabel='Variáveis')

# 10. Média de Preços por Combustível e Tipo de Transmissão
st.subheader('10. Média de Preços por Combustível e Tipo de Transmissão')
df_avg_price = df.groupby(['Fuel_Diesel', 'Transmission_Manual'])['Price'].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df_avg_price, x='Fuel_Diesel', y='Price', hue='Transmission_Manual', palette='Set2', ax=ax)
plot_graph(fig, ax, 'Média de Preços por Combustível e Tipo de Transmissão', xlabel='Tipo de Combustível', ylabel='Preço (US$)')

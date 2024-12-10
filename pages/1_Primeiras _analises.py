import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Carregar o dataset
url_csv = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_novo.csv"
df = pd.read_csv(url_csv)

# Excluir colunas desnecessárias para análise
df = df[['Year', 'KM\'s driven', 'Price', 'Fuel_Diesel', 'Fuel_Petrol', 'Assembly_Local', 'Transmission_Manual']]

# Filtros interativos via sidebar
anos = st.sidebar.multiselect("Selecione o(s) ano(s) para filtrar", df['Year'].unique())
if anos:
    df = df[df['Year'].isin(anos)]

combustivel = st.sidebar.multiselect("Selecione o(s) tipo(s) de combustível", ['Fuel_Diesel', 'Fuel_Petrol'])
if combustivel:
    df = df[df[combustivel].notnull()]

transmissao = st.sidebar.selectbox("Selecione o tipo de transmissão", ['Transmission_Manual'])
if transmissao:
    df = df[df[transmissao].notnull()]

# Configurações comuns de estilo
sns.set(style="whitegrid", palette="muted")
figsize = (10, 6)

# Função para exibir os gráficos de forma modularizada
def plot_graph(fig, ax, title, xlabel=None, ylabel=None, xticks=None, yticks=None, rotation=0):
    ax.set_title(title, fontsize=16)
    if xlabel: ax.set_xlabel(xlabel, fontsize=14)
    if ylabel: ax.set_ylabel(ylabel, fontsize=14)
    if xticks: ax.set_xticklabels(xticks)
    if yticks: ax.set_yticks(yticks)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    st.pyplot(fig)

# 1. Boxplot de Preço por Ano de Fabricação
st.subheader('1. Distribuição de Preços por Ano de Fabricação')
fig, ax = plt.subplots(figsize=figsize)
sns.boxplot(data=df, x='Year', y='Price', palette='Set2', ax=ax)
plot_graph(fig, ax, 'Distribuição de Preços por Ano de Fabricação', xlabel='Ano de Fabricação', ylabel='Preço (R$)', rotation=90)

# 2. Histograma de Quilometragem (KM's driven)
st.subheader('2. Distribuição de Quilometragem dos Carros')
fig, ax = plt.subplots(figsize=figsize)
sns.histplot(df['KM\'s driven'], kde=True, color='blue', bins=30, ax=ax)
plot_graph(fig, ax, 'Distribuição de Quilometragem dos Carros', xlabel='Quilometragem (KM)', ylabel='Frequência')

# 3. Histograma de Preços
st.subheader('3. Distribuição de Preços dos Carros')
fig, ax = plt.subplots(figsize=figsize)
sns.histplot(df['Price'], kde=True, color='green', bins=30, ax=ax)
plot_graph(fig, ax, 'Distribuição de Preços dos Carros', xlabel='Preço (R$)', ylabel='Frequência')

# 4. Boxplot de Preço por Tipo de Combustível
st.subheader('4. Distribuição de Preços por Tipo de Combustível')
fig, ax = plt.subplots(figsize=figsize)
sns.boxplot(data=df, x='Fuel_Diesel', y='Price', palette='Set1', ax=ax)
plot_graph(fig, ax, 'Distribuição de Preços por Tipo de Combustível', xlabel='Tipo de Combustível', ylabel='Preço (R$)', xticks=['Diesel', 'Gasolina'])

# 5. Boxplot de Preço por Local de Montagem
st.subheader('5. Distribuição de Preços por Local de Montagem')
fig, ax = plt.subplots(figsize=figsize)
sns.boxplot(data=df, x='Assembly_Local', y='Price', palette='Set1', ax=ax)
plot_graph(fig, ax, 'Distribuição de Preços por Local de Montagem', xlabel='Local de Montagem', ylabel='Preço (R$)', rotation=45)

# 6. Boxplot de Quilometragem por Tipo de Transmissão
st.subheader('6. Distribuição de Quilometragem por Tipo de Transmissão')
fig, ax = plt.subplots(figsize=figsize)
sns.boxplot(data=df, x='Transmission_Manual', y='KM\'s driven', palette='Set2', ax=ax)
plot_graph(fig, ax, 'Distribuição de Quilometragem por Tipo de Transmissão', xlabel='Tipo de Transmissão', ylabel='Quilometragem (KM)', xticks=['Manual', 'Automática'])

# 7. Gráfico de Dispersão entre Preço e Quilometragem
st.subheader('7. Correlação entre Preço e Quilometragem')
fig, ax = plt.subplots(figsize=figsize)
sns.scatterplot(data=df, x='KM\'s driven', y='Price', color='orange', ax=ax)
plot_graph(fig, ax, 'Correlação entre Preço e Quilometragem', xlabel='Quilometragem (KM)', ylabel='Preço (R$)')

# 8. Distribuição de Preços por Ano (Histograma com KDE)
st.subheader('8. Distribuição de Preços ao Longo dos Anos')
fig, ax = plt.subplots(figsize=figsize)
sns.histplot(df, x='Price', hue='Year', kde=True, palette='viridis', multiple='stack', ax=ax)
plot_graph(fig, ax, 'Distribuição de Preços ao Longo dos Anos', xlabel='Preço (R$)', ylabel='Frequência')

# 9. Matriz de Correlação entre Ano, Quilometragem e Preço
st.subheader('9. Matriz de Correlação')
correlation_matrix = df[['Year', 'KM\'s driven', 'Price']].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Matriz de Correlação', fontsize=16)
plt.tight_layout()
st.pyplot(fig)

# 10. Média de Preços por Combustível e Tipo de Transmissão
st.subheader('10. Média de Preços por Combustível e Tipo de Transmissão')
df_avg_price = df.groupby(['Fuel_Diesel', 'Transmission_Manual'])['Price'].mean().reset_index()
fig, ax = plt.subplots(figsize=figsize)
sns.barplot(data=df_avg_price, x='Fuel_Diesel', y='Price', hue='Transmission_Manual', palette='Set2', ax=ax)
plot_graph(fig, ax, 'Média de Preços por Combustível e Tipo de Transmissão', xlabel='Tipo de Combustível', ylabel='Preço Médio (R$)')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Carregar o dataset
url_csv = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_novo.csv"
df = pd.read_csv(url_csv)

# Excluir colunas desnecessárias para análise
df = df[['Year', 'KM\'s driven', 'Price', 'Fuel_Diesel', 'Fuel_Petrol', 'Assembly_Local', 'Transmission_Manual']]

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

# Definindo o estilo
sns.set(style="whitegrid")

# 1. Boxplot de Preço por Ano de Fabricação
st.subheader('1. Distribuição de Preços por Ano de Fabricação')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='Year', y='Price', palette='Set2', ax=ax)
ax.set_title('Distribuição de Preços por Ano de Fabricação')
plt.xticks(rotation=90)
plt.tight_layout()
st.pyplot(fig)

# 2. Histograma de Quilometragem (KM's driven)
st.subheader('2. Distribuição de Quilometragem dos Carros')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['KM\'s driven'], kde=True, color='blue', bins=30, ax=ax)
ax.set_title('Distribuição de Quilometragem dos Carros')
ax.set_xlabel('Quilometragem (KM)')
ax.set_ylabel('Frequência')
plt.tight_layout()
st.pyplot(fig)

# 3. Histograma de Preços
st.subheader('3. Distribuição de Preços dos Carros')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['Price'], kde=True, color='green', bins=30, ax=ax)
ax.set_title('Distribuição de Preços dos Carros')
ax.set_xlabel('Preço (R$)')
ax.set_ylabel('Frequência')
plt.tight_layout()
st.pyplot(fig)

# 4. Boxplot de Preço por Tipo de Combustível
st.subheader('4. Distribuição de Preços por Tipo de Combustível')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='Fuel_Diesel', y='Price', palette='Set1', ax=ax)
ax.set_title('Distribuição de Preços por Tipo de Combustível')
ax.set_xticklabels(['Diesel', 'Gasolina'])
plt.tight_layout()
st.pyplot(fig)

# 5. Boxplot de Preço por Local de Montagem
st.subheader('5. Distribuição de Preços por Local de Montagem')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='Assembly_Local', y='Price', palette='Set1', ax=ax)
ax.set_title('Distribuição de Preços por Local de Montagem')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# 6. Boxplot de Quilometragem por Tipo de Transmissão
st.subheader('6. Distribuição de Quilometragem por Tipo de Transmissão')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='Transmission_Manual', y='KM\'s driven', palette='Set2', ax=ax)
ax.set_title('Distribuição de Quilometragem por Tipo de Transmissão')
ax.set_xticklabels(['Manual', 'Automática'])
plt.tight_layout()
st.pyplot(fig)

# 7. Gráfico de Dispersão entre Preço e Quilometragem
st.subheader('7. Correlação entre Preço e Quilometragem')
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='KM\'s driven', y='Price', color='orange', ax=ax)
ax.set_title('Correlação entre Preço e Quilometragem')
ax.set_xlabel('Quilometragem (KM)')
ax.set_ylabel('Preço (R$)')
plt.tight_layout()
st.pyplot(fig)

# 8. Distribuição de Preços por Ano (Histograma com KDE)
st.subheader('8. Distribuição de Preços ao Longo dos Anos')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df, x='Price', hue='Year', kde=True, palette='viridis', multiple='stack', ax=ax)
ax.set_title('Distribuição de Preços ao Longo dos Anos')
ax.set_xlabel('Preço (R$)')
ax.set_ylabel('Frequência')
plt.tight_layout()
st.pyplot(fig)

# 9. Matriz de Correlação entre Ano, Quilometragem e Preço
st.subheader('9. Matriz de Correlação')
correlation_matrix = df[['Year', 'KM\'s driven', 'Price']].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Matriz de Correlação')
plt.tight_layout()
st.pyplot(fig)

# 10. Média de Preços por Combustível e Tipo de Transmissão
st.subheader('10. Média de Preços por Combustível e Tipo de Transmissão')
df_avg_price = df.groupby(['Fuel_Diesel', 'Transmission_Manual'])['Price'].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df_avg_price, x='Fuel_Diesel', y='Price', hue='Transmission_Manual', palette='Set2', ax=ax)
ax.set_title('Média de Preços por Combustível e Tipo de Transmissão')
plt.tight_layout()
st.pyplot(fig)

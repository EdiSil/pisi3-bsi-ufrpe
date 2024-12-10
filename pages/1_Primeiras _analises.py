import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Carregar o dataset
url_csv = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_novo.csv"
df = pd.read_csv(url_csv)

# Limpeza de dados: remover ou substituir valores nulos
df = df[['Year', 'KM\'s driven', 'Price', 'Fuel_Diesel', 'Fuel_Petrol', 'Assembly_Local', 'Transmission_Manual']]

# Converter 'Year' para numérico e remover linhas com valores nulos
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df.dropna(subset=['Year', 'Price'], inplace=True)

# Filtros na barra lateral
anos = st.sidebar.multiselect("Selecione o(s) ano(s) para filtrar", df['Year'].unique())
if anos:
    df = df[df['Year'].isin(anos)]

combustivel = st.sidebar.multiselect("Selecione o(s) tipo(s) de combustível", ['Fuel_Diesel', 'Fuel_Petrol'])
if combustivel:
    df = df[df[combustivel].notnull()]

transmissao = st.sidebar.selectbox("Selecione o tipo de transmissão", ['Transmission_Manual'])
if transmissao:
    df = df[df[transmissao].notnull()]

# Convertendo preços para dólares (aproximadamente 1 BRL = 0.19 USD)
df['Price'] = df['Price'] * 0.19

# 1. Distribuição de Preços por Ano de Fabricação
st.subheader('1. Distribuição de Preços por Ano de Fabricação')
try:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Year', y='Price', palette='Set2')
    plt.title('Distribuição de Preços por Ano de Fabricação')
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot()
except ValueError as e:
    st.error(f"Erro ao gerar o gráfico: {str(e)}")
    st.warning("Tentando uma paleta diferente...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Year', y='Price', palette='Set1')
    plt.title('Distribuição de Preços por Ano de Fabricação')
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot()

# 2. Distribuição de Quilometragem dos Carros
st.subheader('2. Distribuição de Quilometragem dos Carros')
plt.figure(figsize=(10, 6))
sns.histplot(df['KM\'s driven'], kde=True, color='blue', bins=30)
plt.title('Distribuição de Quilometragem dos Carros')
plt.xlabel('Quilometragem (KM)')
plt.ylabel('Frequência')
plt.tight_layout()
st.pyplot()

# 3. Distribuição de Preços dos Carros
st.subheader('3. Distribuição de Preços dos Carros')
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], kde=True, color='green', bins=30)
plt.title('Distribuição de Preços dos Carros')
plt.xlabel('Preço (USD)')
plt.ylabel('Frequência')
plt.tight_layout()
st.pyplot()

# 4. Distribuição de Preços por Tipo de Combustível
st.subheader('4. Distribuição de Preços por Tipo de Combustível')
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Fuel_Diesel', y='Price', palette='Set1')
plt.xticks([0, 1], ['Diesel', 'Gasolina'])
plt.title('Distribuição de Preços por Tipo de Combustível')
plt.tight_layout()
st.pyplot()

# 5. Distribuição de Preços por Local de Montagem
st.subheader('5. Distribuição de Preços por Local de Montagem')
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Assembly_Local', y='Price', palette='Set1')
plt.xticks(rotation=45)
plt.title('Distribuição de Preços por Local de Montagem')
plt.tight_layout()
st.pyplot()

# 6. Distribuição de Quilometragem por Tipo de Transmissão
st.subheader('6. Distribuição de Quilometragem por Tipo de Transmissão')
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Transmission_Manual', y='KM\'s driven', palette='Set2')
plt.xticks([0, 1], ['Manual', 'Automática'])
plt.title('Distribuição de Quilometragem por Tipo de Transmissão')
plt.tight_layout()
st.pyplot()

# 7. Correlação entre Preço e Quilometragem
st.subheader('7. Correlação entre Preço e Quilometragem')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='KM\'s driven', y='Price', color='orange')
plt.title('Correlação entre Preço e Quilometragem')
plt.xlabel('Quilometragem (KM)')
plt.ylabel('Preço (USD)')
plt.tight_layout()
st.pyplot()

# 8. Distribuição de Preços ao Longo dos Anos
st.subheader('8. Distribuição de Preços ao Longo dos Anos')
plt.figure(figsize=(10, 6))
sns.histplot(df, x='Price', hue='Year', kde=True, palette='viridis', multiple='stack')
plt.title('Distribuição de Preços ao Longo dos Anos')
plt.xlabel('Preço (USD)')
plt.ylabel('Frequência')
plt.tight_layout()
st.pyplot()

# 9. Matriz de Correlação
st.subheader('9. Matriz de Correlação entre Ano, Quilometragem e Preço')
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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Função para carregar e limpar os dados
@st.cache_data
def load_and_clean_data():
    # Carregar o dataset
    url_csv = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_novo.csv"
    df = pd.read_csv(url_csv)

    # Limpeza de dados: manter apenas as colunas de interesse
    df = df[['Year', 'KM\'s driven', 'Price', 'Fuel_Diesel', 'Fuel_Petrol', 'Assembly_Local', 'Transmission_Manual']]

    # Limpeza de valores nulos e conversão de tipo de dados
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df.dropna(subset=['Year', 'Price'], inplace=True)
    
    # Convertendo Preço para dólares
    df['Price'] = df['Price'] * 0.19  # Aproximadamente 1 BRL = 0.19 USD

    return df

# Carregar e preparar os dados
df = load_and_clean_data()

# Filtros na barra lateral para permitir interatividade
anos = st.sidebar.multiselect("Selecione o(s) ano(s)", df['Year'].unique())
combustivel = st.sidebar.multiselect("Selecione o(s) combustível(is)", ['Fuel_Diesel', 'Fuel_Petrol'])
transmissao = st.sidebar.selectbox("Selecione a transmissão", ['Transmission_Manual'])

# Aplicando os filtros
if anos:
    df = df[df['Year'].isin(anos)]

if combustivel:
    df = df[df[combustivel].notnull()]

if transmissao:
    df = df[df[transmissao].notnull()]

# Função para plotar gráficos
def plot_graph(func, *args, **kwargs):
    # Criando uma nova figura
    fig, ax = plt.subplots(figsize=(10, 6))
    func(*args, **kwargs)
    plt.tight_layout()
    st.pyplot(fig)  # Passando o objeto figura para o st.pyplot()

# 1. Distribuição de Preços por Ano de Fabricação
st.subheader('Distribuição de Preços por Ano de Fabricação')
plot_graph(sns.boxplot, data=df, x='Year', y='Price', palette='Set2')
plt.title('Distribuição de Preços por Ano')

# 2. Distribuição de Quilometragem
st.subheader('Distribuição de Quilometragem')
plot_graph(sns.histplot, df['KM\'s driven'], kde=True, color='blue', bins=30)
plt.title('Distribuição de Quilometragem dos Carros')

# 3. Distribuição de Preços
st.subheader('Distribuição de Preços')
plot_graph(sns.histplot, df['Price'], kde=True, color='green', bins=30)
plt.title('Distribuição de Preços dos Carros')

# 4. Distribuição de Preços por Tipo de Combustível
st.subheader('Distribuição de Preços por Tipo de Combustível')
plot_graph(sns.boxplot, data=df, x='Fuel_Diesel', y='Price', palette='Set1')
plt.title('Distribuição de Preços por Combustível')

# 5. Distribuição de Preços por Local de Montagem
st.subheader('Distribuição de Preços por Local de Montagem')
plot_graph(sns.boxplot, data=df, x='Assembly_Local', y='Price', palette='Set1')
plt.xticks(rotation=45)
plt.title('Distribuição de Preços por Local de Montagem')

# 6. Distribuição de Quilometragem por Tipo de Transmissão
st.subheader('Distribuição de Quilometragem por Tipo de Transmissão')
plot_graph(sns.boxplot, data=df, x='Transmission_Manual', y='KM\'s driven', palette='Set2')
plt.title('Distribuição de Quilometragem por Tipo de Transmissão')

# 7. Correlação entre Preço e Quilometragem
st.subheader('Correlação entre Preço e Quilometragem')
plot_graph(sns.scatterplot, data=df, x='KM\'s driven', y='Price', color='orange')
plt.title('Correlação entre Preço e Quilometragem')

# 8. Distribuição de Preços ao Longo dos Anos
st.subheader('Distribuição de Preços ao Longo dos Anos')
plot_graph(sns.histplot, df, x='Price', hue='Year', kde=True, palette='viridis', multiple='stack')
plt.title('Distribuição de Preços ao Longo dos Anos')

# 9. Matriz de Correlação
st.subheader('Matriz de Correlação')
corr_matrix = df[['Year', 'KM\'s driven', 'Price']].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
plt.title('Matriz de Correlação entre Ano, Quilometragem e Preço')
st.pyplot(fig)

# 10. Média de Preços por Combustível e Tipo de Transmissão
st.subheader('Média de Preços por Combustível e Tipo de Transmissão')
df_avg_price = df.groupby(['Fuel_Diesel', 'Transmission_Manual'])['Price'].mean().reset_index()
plot_graph(sns.barplot, data=df_avg_price, x='Fuel_Diesel', y='Price', hue='Transmission_Manual', palette='Set2')
plt.title('Média de Preços por Combustível e Tipo de Transmissão')

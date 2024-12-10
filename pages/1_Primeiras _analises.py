import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Carregar o dataset diretamente do URL
url_csv = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_novo.csv"
df = pd.read_csv(url_csv)

# Verificar as colunas disponíveis no dataset
st.write("Colunas disponíveis no dataset:", df.columns)

# Exibir os primeiros registros para entender o formato dos dados
st.write("Primeiras linhas dos dados carregados:", df.head())

# Limpeza de dados nulos para as colunas relevantes
df = df.dropna(subset=['Year', 'KM\'s driven', 'Price', 'Fuel_Diesel', 'Fuel_Petrol', 'Assembly_Local', 'Transmission_Manual'])

# Adicionar título à aplicação Streamlit
st.title('Análise de Carros Usados - Primeiras Análises')

# Filtros no sidebar para selecionar combustíveis e tipo de transmissão
fuel_type = st.sidebar.multiselect("Selecione o tipo de combustível", df['Fuel_Diesel'].unique())
transmission_type = st.sidebar.multiselect("Selecione o tipo de transmissão", df['Transmission_Manual'].unique())

# Filtragem de dados
df_filtered = df[df['Fuel_Diesel'].isin(fuel_type)] if fuel_type else df
df_filtered = df_filtered[df_filtered['Transmission_Manual'].isin(transmission_type)] if transmission_type else df_filtered

st.write(f"Dados filtrados para os combustíveis: {', '.join(fuel_type) if fuel_type else 'Todos os combustíveis'}")
st.write(f"Dados filtrados para as transmissões: {', '.join(transmission_type) if transmission_type else 'Todas as transmissões'}")

# Boxplot de Variação do Preço Pelo Ano de Fabricação
st.subheader("Boxplot de Variação do Preço Pelo Ano de Fabricação")
sns.boxplot(data=df_filtered, x='Year', y='Price', palette='Set2')
plt.xticks(rotation=90)
st.pyplot()

# Boxplot de Variação da Quilometragem pelo Ano de Fabricação
st.subheader("Boxplot de Variação da Quilometragem Pelo Ano de Fabricação")
sns.boxplot(data=df_filtered, x='Year', y='KM\'s driven', palette='Set2')
st.pyplot()

# Boxplot de Variação do Preço Pelo Tipo de Combustível (Diesel e Gasolina)
st.subheader("Boxplot de Variação do Preço Pelo Tipo de Combustível")
sns.boxplot(data=df_filtered, x='Fuel_Diesel', y='Price', palette='Set1')
st.pyplot()

# Boxplot de Variação da Quilometragem Pelo Tipo de Combustível
st.subheader("Boxplot de Variação da Quilometragem Pelo Tipo de Combustível")
sns.boxplot(data=df_filtered, x='Fuel_Diesel', y='KM\'s driven', palette='Set1')
st.pyplot()

# Boxplot de Variação do Preço Pelo Local de Montagem
st.subheader("Boxplot de Variação do Preço Pelo Local de Montagem")
sns.boxplot(data=df_filtered, x='Assembly_Local', y='Price', palette='Set1')
plt.xticks(rotation=90)
st.pyplot()

# Boxplot de Variação do Preço Pelo Tipo de Transmissão (Manual)
st.subheader("Boxplot de Variação do Preço Pelo Tipo de Transmissão")
sns.boxplot(data=df_filtered, x='Transmission_Manual', y='Price', palette='Set1')
st.pyplot()

# Histograma de Preço vs Quilometragem
st.subheader("Histograma - Variação de Preço com Quilometragem")
sns.histplot(df_filtered, x='KM\'s driven', hue='Price', multiple="stack", kde=True, palette='coolwarm')
st.pyplot()

# Exibir os dados filtrados para o usuário
st.subheader("Dados Filtrados")
st.write(df_filtered)

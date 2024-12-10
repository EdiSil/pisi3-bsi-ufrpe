import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Carregar o dataset diretamente do URL
url_csv = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_novo.csv"
df = pd.read_csv(url_csv)

# Verificar as colunas disponíveis
st.write("Colunas disponíveis no dataset:", df.columns)

# Limpeza de dados nulos para as colunas relevantes
df = df.dropna(subset=['Year', 'KM\'s driven', 'Price', 'Fuel_Diesel', 'Fuel_Petrol', 'Assembly_Local', 'Transmission_Manual'])

# Adicionar título à aplicação Streamlit
st.title('Análise de Carros Usados - Primeiras Análises')

# Filtros no sidebar para selecionar marcas
marcas = st.sidebar.multiselect("Selecione as marcas para filtrar", df['marca'].unique())
df_filtered = df[df['marca'].isin(marcas)] if marcas else df

# Exibição dos dados filtrados
st.write(f"Dados filtrados para as marcas: {', '.join(marcas) if marcas else 'Todas as marcas'}")

# Boxplot da Marca vs Quilometragem
st.subheader("Boxplot da Marca x Quilometragem")
sns.boxplot(data=df_filtered, x='marca', y='KM\'s driven', palette='Set2')
plt.xticks(rotation=90)
st.pyplot()

# Histograma da Variação do Preço pelo Ano de Fabricação
st.subheader("Histograma - Variação do Preço Pelo Ano de Fabricação")
sns.histplot(df_filtered, x='Year', hue='Price', multiple="stack", palette='coolwarm', kde=True)
st.pyplot()

# Boxplot de Variação do Preço Pelo Modelo do Veículo
st.subheader("Boxplot de Variação do Preço Pelo Modelo do Veículo")
sns.boxplot(data=df_filtered, x='modelo', y='Price', palette='Set1')
plt.xticks(rotation=90)
st.pyplot()

# Boxplot de Variação do Preço Pelo Combustível (Diesel x Gasolina)
st.subheader("Boxplot - Variação do Preço Pelo Combustível")
sns.boxplot(data=df_filtered, x='Fuel_Diesel', y='Price', palette='Set1')
st.pyplot()

# Boxplot de Variação do Preço Pelo Tipo de Transmissão
st.subheader("Boxplot - Variação do Preço Pelo Tipo de Transmissão")
sns.boxplot(data=df_filtered, x='Transmission_Manual', y='Price', palette='Set1')
st.pyplot()

# Boxplot de Variação da Quilometragem pelo Ano de Fabricação
st.subheader("Boxplot - Variação da Quilometragem Pelo Ano de Fabricação")
sns.boxplot(data=df_filtered, x='Year', y='KM\'s driven', palette='Set2')
st.pyplot()

# Boxplot de Variação da Quilometragem pelo Preço
st.subheader("Boxplot - Variação da Quilometragem Pelo Preço")
sns.boxplot(data=df_filtered, x='Price', y='KM\'s driven', palette='Set2')
st.pyplot()

# Mostrar os dados filtrados para o usuário
st.subheader("Dados Filtrados")
st.write(df_filtered)

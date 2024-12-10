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

# Filtros para interação do usuário
st.sidebar.header("Filtros")
year_filter = st.sidebar.slider("Selecione o Ano de Fabricação", min_value=int(df['Year'].min()), max_value=int(df['Year'].max()), value=(int(df['Year'].min()), int(df['Year'].max())))
df_filtered = df[(df['Year'] >= year_filter[0]) & (df['Year'] <= year_filter[1])]

# Boxplot: Preço x Ano de Fabricação
st.subheader("Boxplot: Preço x Ano de Fabricação")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_filtered, x='Year', y='Price', palette='Set2')
plt.xticks(rotation=45)
plt.title('Distribuição de Preços ao Longo dos Anos')
plt.xlabel('Ano de Fabricação')
plt.ylabel('Preço')
st.pyplot(plt.gcf())

# Histogram: Preço vs Quilometragem (KM's driven)
st.subheader("Histograma: Preço vs Quilometragem")
plt.figure(figsize=(12, 6))
sns.histplot(data=df_filtered, x='KM\'s driven', y='Price', bins=30, kde=True, color='blue')
plt.title('Variação do Preço com Base na Quilometragem')
plt.xlabel('Quilometragem (KM\'s driven)')
plt.ylabel('Preço')
st.pyplot(plt.gcf())

# Boxplot: Variação do Preço com Base no Preço Condition
st.subheader("Boxplot: Preço x Condição do Preço")
# Corrigir erro relacionado ao 'Price Condition'. Certifique-se de que a coluna não contém valores NaN ou inconsistentes
df_filtered['Price Condition'] = df_filtered['Price Condition'].fillna('Desconhecido')
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_filtered, x='Price Condition', y='Price', palette='Set1')
plt.title('Distribuição de Preços de Acordo com a Condição do Preço')
plt.xlabel('Condição do Preço')
plt.ylabel('Preço')
st.pyplot(plt.gcf())

# Boxplot: Variação do Preço por Tipo de Combustível (Diesel vs. Petrol)
st.subheader("Boxplot: Preço x Tipo de Combustível")
# Corrigir erro similar: garantir que as colunas 'Fuel_Diesel' e 'Fuel_Petrol' tenham valores válidos
df_filtered['Fuel_Diesel'] = df_filtered['Fuel_Diesel'].fillna('Desconhecido')
df_filtered['Fuel_Petrol'] = df_filtered['Fuel_Petrol'].fillna('Desconhecido')

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_filtered, x='Fuel_Diesel', y='Price', palette='Set2')
sns.boxplot(data=df_filtered, x='Fuel_Petrol', y='Price', palette='Set2')
plt.title('Distribuição de Preços por Tipo de Combustível')
plt.xlabel('Tipo de Combustível')
plt.ylabel('Preço')
st.pyplot(plt.gcf())

# Boxplot: Variação do Preço pela Localização de Montagem
st.subheader("Boxplot: Preço x Local de Montagem")
# Corrigir erro semelhante
df_filtered['Assembly_Local'] = df_filtered['Assembly_Local'].fillna('Desconhecido')

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_filtered, x='Assembly_Local', y='Price', palette='Set3')
plt.title('Distribuição de Preços por Local de Montagem')
plt.xlabel('Local de Montagem')
plt.ylabel('Preço')
st.pyplot(plt.gcf())

# Boxplot: Variação do Preço pela Transmissão Manual
st.subheader("Boxplot: Preço x Transmissão Manual")
# Corrigir erro semelhante
df_filtered['Transmission_Manual'] = df_filtered['Transmission_Manual'].fillna('Desconhecido')

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_filtered, x='Transmission_Manual', y='Price', palette='Set3')
plt.title('Distribuição de Preços por Tipo de Transmissão')
plt.xlabel('Transmissão Manual')
plt.ylabel('Preço')
st.pyplot(plt.gcf())

# Análise interativa do painel
st.markdown("### Utilize os filtros para personalizar a análise de acordo com o seu interesse!")

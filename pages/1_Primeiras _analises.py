import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Configurar tema do Seaborn
sns.set_theme(style="whitegrid")

# Função para carregar e limpar os dados
@st.cache_data
def load_and_clean_data():
    url_csv = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_novo.csv"
    try:
        df = pd.read_csv(url_csv)
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame()  # Retorna um DataFrame vazio em caso de erro

    # Selecionar colunas relevantes
    df = df[['Year', 'KM\'s driven', 'Price', 'Fuel_Diesel', 'Fuel_Petrol', 'Assembly_Local', 'Transmission_Manual']]

    # Converter colunas para tipos apropriados
    for col in ['Year', 'Price', 'KM\'s driven']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filtrar valores válidos
    df = df[(df['Year'] > 1900) & (df['Year'] <= 2024) & (df['Price'] > 0)]
    df = df.dropna(subset=['Year', 'Price', 'KM\'s driven'])

    # Converter preços para dólares (1 BRL = 0.19 USD)
    df['Price'] *= 0.19

    return df

# Função para plotar gráficos
def plot_graph(func, *args, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 6))
    func(*args, **kwargs, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

# Carregar e limpar dados
df = load_and_clean_data()

# Verificar se o DataFrame contém dados válidos
if df.empty:
    st.error("Nenhum dado disponível após a limpeza. Verifique os filtros.")
else:
    # Filtros na barra lateral
    st.sidebar.header("Filtros")
    
    # Filtro por Ano de Fabricação
    anos = st.sidebar.multiselect("Selecione o(s) ano(s)", sorted(df['Year'].unique()))
    if anos:
        df = df[df['Year'].isin(anos)]

    # Filtro por Tipo de Combustível
    combustivel = st.sidebar.multiselect("Selecione o tipo de combustível", ['Fuel_Diesel', 'Fuel_Petrol'])
    if combustivel:
        df = df[df[combustivel].any(axis=1)]

    # Filtro por Tipo de Transmissão
    transmissao = st.sidebar.selectbox("Selecione o tipo de transmissão", ['Manual'])
    if transmissao:
        df = df[df['Transmission_Manual'] == 1]

    # Verificar novamente se o DataFrame contém dados após filtros
    if df.empty:
        st.error("Nenhum dado disponível após aplicar os filtros selecionados.")
    else:
        # 1. Distribuição de Preços por Ano de Fabricação
        st.subheader("Distribuição de Preços por Ano de Fabricação")
        plot_graph(sns.boxplot, data=df, x='Year', y='Price', palette='Set2')
        plt.title("Distribuição de Preços por Ano de Fabricação")

        # 2. Distribuição de Quilometragem dos Carros
        st.subheader("Distribuição de Quilometragem dos Carros")
        plot_graph(sns.histplot, data=df, x="KM's driven", kde=True, color='blue', bins=30)
        plt.title("Distribuição de Quilometragem dos Carros")
        plt.xlabel("Quilometragem (KM)")
        plt.ylabel("Frequência")

        # 3. Distribuição de Preços dos Carros
        st.sub

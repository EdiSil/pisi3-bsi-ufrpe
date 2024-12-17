import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Configurar tema do Seaborn
sns.set_theme(style="whitegrid")

# Função para carregar e limpar os dados
@st.cache_data
def load_and_clean_data():
    url_csv = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_dataset00.csv"
    df = pd.read_csv(url_csv)

    # Selecionar colunas relevantes
    df = df[['Year', 'KM\'s driven', 'Price', 'Fuel_Diesel', 'Fuel_Petrol', 'Assembly_Local', 'Transmission_Manual']]

    # Converter colunas para tipos apropriados
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['KM\'s driven'] = pd.to_numeric(df['KM\'s driven'], errors='coerce')

    # Filtrar valores válidos
    df = df[(df['Year'] >= 1994) & (df['Year'] <= 2023)]  # Filtrar anos válidos
    df = df[df['Price'] > 0]  # Filtrar preços positivos
    df = df.dropna(subset=['Year', 'Price', 'KM\'s driven'])  # Remover linhas com valores ausentes

    # Converter preços para dólares (1 BRL = 0.19 USD)
    df['Price'] = df['Price'] * 0.19

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
        combustiveis_cols = [col for col in combustivel if col in df.columns]
        if combustiveis_cols:
            df = df[df[combustiveis_cols].any(axis=1)]

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
        try:
            plot_graph(sns.boxplot, data=df, x='Year', y='Price', palette='Set2')
            plt.title("Distribuição de Preços por Ano de Fabricação")
        except Exception as e:
            st.error(f"Erro ao gerar gráfico: {e}")

        # 2. Distribuição de Quilometragem dos Carros
        st.subheader("Distribuição da Quilometragem dos Carros")
        plot_graph(sns.histplot, data=df, x="KM's driven", kde=True, color='blue', bins=30)
        plt.title("Distribuição de Quilometragem dos Carros")
        plt.xlabel("Quilometragem (KM)")
        plt.ylabel("Frequência")

        # 3. Distribuição de Preços dos Carros
        st.subheader("Distribuição dos Preços dos Carros")
        plot_graph(sns.histplot, data=df, x='Price', kde=True, color='green', bins=30)
        plt.title("Distribuição de Preços dos Carros")
        plt.xlabel("Preço (USD)")
        plt.ylabel("Frequência")

        # 4. Distribuição de Preços por Tipo de Combustível
        st.subheader("Distribuição dos Preços por Tipo de Combustível")
        combustivel_data = df.melt(id_vars=['Price'], value_vars=['Fuel_Diesel', 'Fuel_Petrol'],
                                   var_name='Fuel_Type', value_name='Is_Fuel_Type')
        combustivel_data = combustivel_data[combustivel_data['Is_Fuel_Type'] == 1]
        plot_graph(sns.boxplot, data=combustivel_data, x='Fuel_Type', y='Price', palette='Set1')
        plt.title("Distribuição de Preços por Tipo de Combustível")
        plt.xticks([0, 1], ['Diesel', 'Petrol'])

        # 5. Correlação entre Preço e Quilometragem
        st.subheader("Correlação entre Preço e Quilometragem")
        plot_graph(sns.scatterplot, data=df, x="KM's driven", y='Price', color='orange')
        plt.title("Correlação entre Preço e Quilometragem")
        plt.xlabel("Quilometragem (KM)")
        plt.ylabel("Preço (USD)")

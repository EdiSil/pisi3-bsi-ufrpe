import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Função para carregar e limpar os dados
@st.cache_data
def load_and_clean_data():
    url_csv = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_novo.csv"
    df = pd.read_csv(url_csv)

    # Filtrar colunas relevantes
    df = df[['Year', 'KM\'s driven', 'Price', 'Fuel_Diesel', 'Fuel_Petrol', 'Assembly_Local', 'Transmission_Manual']]

    # Limpar dados ausentes ou inválidos
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df[(df['Year'] > 1900) & (df['Year'] <= 2024)]  # Exemplo de intervalo aceitável para anos
    df = df[df['Price'] > 0]  # Preço deve ser maior que 0

    # Converter para dólares
    df['Price'] = df['Price'] * 0.19  # Conversão BRL -> USD

    return df

# Carregar e limpar os dados
df = load_and_clean_data()

# Filtrar valores inválidos antes de gerar gráficos
if df.empty:
    st.error("Nenhum dado disponível após a limpeza. Verifique os filtros.")
else:
    # Função para plotar gráficos
    def plot_graph(func, *args, **kwargs):
        fig, ax = plt.subplots(figsize=(10, 6))
        func(*args, **kwargs, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)

    # 1. Distribuição de Preços por Ano de Fabricação
    st.subheader("Distribuição de Preços por Ano de Fabricação")
    try:
        plot_graph(sns.boxplot, data=df, x='Year', y='Price', palette='Set2')
        plt.title("Distribuição de Preços por Ano")
    except ValueError as e:
        st.error(f"Erro ao gerar gráfico: {e}")

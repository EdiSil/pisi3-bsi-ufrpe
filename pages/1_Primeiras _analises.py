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
    df = pd.read_csv(url_csv)

    # Selecionar colunas relevantes
    df = df[['Year', 'KM\'s driven', 'Price', 'Fuel_Diesel', 'Fuel_Petrol', 'Assembly_Local', 'Transmission_Manual']]

    # Converter colunas para tipos apropriados
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['KM\'s driven'] = pd.to_numeric(df['KM\'s driven'], errors='coerce')

    # Filtrar valores válidos
    df = df[(df['Year'] > 1900) & (df['Year'] <= 2024)]  # Filtrar anos válidos
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
        plot_graph(sns.boxplot, data=df, x='Year', y='Price', palette='Set2')

        # 2. Distribuição de Quilometragem dos Carros
        st.subheader("Distribuição de Quilometragem dos Carros")
        plot_graph(sns.histplot, data=df, x="KM's driven", kde=True, color='blue', bins=30)

        # 3. Distribuição de Preços por Tipo de Combustível
        st.subheader("Distribuição de Preços por Tipo de Combustível")
        combustivel_data = df.melt(id_vars=['Price'], value_vars=['Fuel_Diesel', 'Fuel_Petrol'],
                                   var_name='Fuel_Type', value_name='Is_Fuel_Type')
        combustivel_data = combustivel_data[combustivel_data['Is_Fuel_Type'] == 1]
        plot_graph(sns.boxplot, data=combustivel_data, x='Fuel_Type', y='Price', palette='Set1')

        # 4. Relação entre Preço e Ano por Tipo de Combustível
        st.subheader("Relação entre Preço e Ano por Tipo de Combustível")
        plot_graph(sns.scatterplot, data=combustivel_data, x='Year', y='Price', hue='Fuel_Type', palette='cool')

        # 5. Correlação entre todas as variáveis numéricas
        st.subheader("Mapa de Correlação")
        corr = df[['Year', 'Price', 'KM\'s driven']].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        plt.title("Correlação entre Variáveis")
        st.pyplot(fig)

        # 6. Frequência por Tipo de Combustível e Transmissão
        st.subheader("Frequência por Tipo de Combustível e Transmissão")
        df['Fuel_Type'] = df[['Fuel_Diesel', 'Fuel_Petrol']].idxmax(axis=1)
        transmission_count = df.groupby(['Fuel_Type', 'Transmission_Manual']).size().reset_index(name='Count')
        transmission_count['Transmission'] = transmission_count['Transmission_Manual'].apply(lambda x: 'Manual' if x == 1 else 'Automático')
        plot_graph(sns.barplot, data=transmission_count, x='Fuel_Type', y='Count', hue='Transmission', palette='Set3')

        # 7. Distribuição de Quilometragem por Ano
        st.subheader("Distribuição de Quilometragem por Ano")
        plot_graph(sns.boxplot, data=df, x='Year', y="KM's driven", palette='Set2')

        # 8. Relação entre Preço e Quilometragem
        st.subheader("Correlação entre Preço e Quilometragem")
        plot_graph(sns.scatterplot, data=df, x="KM's driven", y='Price', color='orange')

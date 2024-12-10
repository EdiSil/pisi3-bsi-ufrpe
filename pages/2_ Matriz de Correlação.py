import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Configuração do tema do Seaborn
sns.set_theme(style="whitegrid")

# Função para carregar os dados diretamente da URL
@st.cache_data
def carregar_dados(url):
    """
    Carrega os dados de um arquivo CSV hospedado em uma URL.

    Args:
        url (str): URL do arquivo CSV.

    Returns:
        pd.DataFrame: Dataset carregado.
    """
    try:
        dados = pd.read_csv(url)
        return dados
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame()

# Função para gerar e exibir a matriz de correlação
def exibir_matriz_correlacao(data):
    """
    Exibe a matriz de correlação e o heatmap das variáveis numéricas.

    Args:
        data (pd.DataFrame): Dataset a ser analisado.
    """
    st.write("### Matriz de Correlação")
    
    # Selecionar apenas as colunas numéricas
    colunas_numericas = data.select_dtypes(include=["number"])

    if colunas_numericas.empty:
        st.error("O dataset não possui colunas numéricas para gerar uma matriz de correlação.")
        return

    # Calcular a matriz de correlação
    matriz_corr = colunas_numericas.corr()

    # Plotar o heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matriz_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="black",
        square=True,
        ax=ax
    )
    ax.set_title("Heatmap da Matriz de Correlação", fontsize=16)

    # Exibir o gráfico no Streamlit
    st.pyplot(fig)

# Função principal para execução do app
def main():
    """
    Função principal do aplicativo Streamlit.
    """
    st.title("Matriz de Correlação de Dados")

    # URL do dataset
    url_csv = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_novo.csv"

    # Carregar os dados
    data = carregar_dados(url_csv)

    if not data.empty:
        st.write("### Primeiras Linhas do Dataset")
        st.write(data.head())  # Exibir as primeiras linhas do dataset

        # Exibir a matriz de correlação
        exibir_matriz_correlacao(data)
    else:
        st.error("Não foi possível carregar os dados.")

# Executar o app
if __name__ == "__main__":
    main()

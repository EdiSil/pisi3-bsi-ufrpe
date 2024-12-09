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
def exibir_matriz_correlacao(data, colunas_selecionadas):
    """
    Exibe a matriz de correlação e o heatmap das variáveis numéricas selecionadas pelo usuário.

    Args:
        data (pd.DataFrame): Dataset a ser analisado.
        colunas_selecionadas (list): Lista de colunas selecionadas pelo usuário para análise.
    """
    st.write("### Matriz de Correlação")

    # Filtrar o dataset pelas colunas selecionadas
    data_filtrada = data[colunas_selecionadas]

    # Calcular a matriz de correlação
    matriz_corr = data_filtrada.corr()

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

    # Explicação sobre a matriz de correlação
    st.write("""
    ### Principais Observações:

    **Year e Price**: A correlação positiva de 0.68 indica uma relação direta moderada. Ou seja, à medida que o ano aumenta, o preço tende a aumentar também.

    **Year e KM's driven**: A correlação negativa de -0.39 sugere uma relação inversa fraca. Isto implica que carros mais novos tendem a ter menos quilometragem.

    **KM's driven e Price**: A correlação não está visível na matriz, mas podemos inferir que, devido à relação inversa entre "Year" e "KM's driven" e uma relação direta entre "Year" e "Price", a correlação entre "KM's driven" e "Price" provavelmente é negativa.
    """)

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
        # Exibir opções de seleção para as colunas
        colunas_disponiveis = data.select_dtypes(include=["number"]).columns.tolist()

        # Permitir que o usuário selecione múltiplas colunas
        colunas_selecionadas = st.multiselect(
            "Selecione as colunas:",
            colunas_disponiveis,
            default=colunas_disponiveis  # Definir as colunas por padrão como todas numericas
        )

        # Verificar se o usuário selecionou colunas
        if colunas_selecionadas:
            # Exibir a matriz de correlação com as colunas selecionadas
            exibir_matriz_correlacao(data, colunas_selecionadas)
        else:
            st.error("Por favor, selecione ao menos uma coluna.")
    else:
        st.error("Não foi possível carregar os dados.")

# Executar o app
if __name__ == "__main__":
    main()

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Função para carregar os dados
def carregar_arquivo():
    url_csv = "https://github.com/EdiSil/pisi3-bsi-ufrpe/blob/main/data/OLX_cars_dataset00.csv"  # URL para exemplo
    try:
        data = pd.read_csv(url_csv)
        return data
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

# Função para exibir a matriz de correlação e o heatmap
def matriz_correlacao_e_heatmap(data):
    # Filtrar apenas as colunas numéricas
    data_numerica = data.select_dtypes(include=['float64', 'int64'])

    # Tratar valores ausentes: podemos preencher com a média ou remover as linhas
    data_numerica = data_numerica.fillna(data_numerica.mean())  # Preencher NaN com a média

    # Calcular a matriz de correlação
    corr_matrix = data_numerica.corr()

    # Configurar o gráfico
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Matriz de Correlação e Heatmap", fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    # Exibir o gráfico no Streamlit
    st.pyplot(plt)

# Função principal do aplicativo Streamlit
def main():
    st.title("Análise de Correlação com Heatmap")
    
    # Carregar os dados
    data = carregar_arquivo()

    if data is not None:
        # Exibir as primeiras linhas dos dados carregados
        st.write("### Primeiras Linhas dos Dados Carregados:")
        st.write(data.head())

        # Exibir a matriz de correlação e o heatmap
        st.write("### Matriz de Correlação")
        matriz_correlacao_e_heatmap(data)

        # Exibir a matriz de correlação como uma tabela
        st.write("### Tabela da Matriz de Correlação")
        st.write(data.corr())

# Executar o aplicativo Streamlit
if __name__ == "__main__":
    main()

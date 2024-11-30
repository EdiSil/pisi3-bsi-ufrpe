import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Função para carregar os dados diretamente da URL do GitHub
def carregar_dados():
    """
    Esta função carrega os dados diretamente do arquivo CSV hospedado no GitHub.
    """
    url_csv = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_novo.csv"
    try:
        # Carregar os dados do CSV
        data = pd.read_csv(url_csv)
        return data
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None

# Função para gerar a matriz de correlação e heatmap
def plotar_matriz_correlacao(data):
    """
    Esta função gera a matriz de correlação entre todas as variáveis numéricas do dataset
    e exibe um heatmap para facilitar a visualização das correlações.
    """
    st.write("### Matriz de Correlação entre Variáveis")
    
    # Selecionar apenas as colunas numéricas
    data_numerico = data.select_dtypes(include=["number"])

    if data_numerico.empty:
        st.error("O dataset não contém colunas numéricas para calcular a correlação.")
        return

    # Calculando a matriz de correlação
    corr = data_numerico.corr()

    # Configuração do tamanho da figura para o heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Gerando o heatmap da matriz de correlação
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, linecolor='black', square=True, ax=ax)
    
    # Adicionando título ao gráfico
    ax.set_title('Matriz de Correlação', fontsize=16)
    
    # Exibindo o gráfico no Streamlit
    st.pyplot(fig)

# Função principal para exibir todas as visualizações
def main():
    # Carregar os dados
    data = carregar_dados()

    # Verificando se os dados foram carregados corretamente
    if data is not None:
        st.title("Análise Exploratória de Dados - OLX Carros")
        
        # Mostrar as primeiras linhas do dataset
        st.write("### Primeiras Linhas do Dataset:")
        st.write(data.head())  # Exibe as primeiras linhas do dataset
        
        # Visualizar a matriz de correlação
        plotar_matriz_correlacao(data)
    else:
        st.error("Dados não carregados. Verifique a disponibilidade do dataset.")

# Executando a função principal
if __name__ == "__main__":
    main()

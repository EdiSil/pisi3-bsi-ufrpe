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

# Função para visualizar a distribuição de preços
def plotar_distribuicao_precos(data):
    """
    Esta função gera um histograma da distribuição dos preços dos carros e também adiciona a curva de densidade
    (KDE - Kernel Density Estimation) para visualizar melhor a distribuição.
    """
    st.write("### Distribuição de Preços")
    
    # Configuração do tamanho da figura para melhor visualização
    fig, ax = plt.subplots(figsize=(10, 6))

    # Gerando o histograma com a curva KDE
    sns.histplot(data['Price'], kde=True, color='blue', bins=30, ax=ax)
    
    # Adicionando título e rótulos aos eixos
    ax.set_title('Distribuição de Preços de Carros', fontsize=16)
    ax.set_xlabel('Preço (R$)', fontsize=12)
    ax.set_ylabel('Frequência', fontsize=12)
    
    # Exibindo o gráfico no Streamlit
    st.pyplot(fig)

# Função para comparar KM driven com o preço
def plotar_relacao_km_preco(data):
    """
    Esta função gera um gráfico de dispersão (scatter plot) que compara a quantidade de quilômetros rodados
    (KM driven) com o preço do carro. Este gráfico ajuda a entender a relação entre essas duas variáveis.
    """
    st.write("### Relação entre KM driven e Preço")
    
    # Configuração do tamanho da figura para o gráfico de dispersão
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Gerando o gráfico de dispersão
    sns.scatterplot(x=data['KM\'s driven'], y=data['Price'], color='green', alpha=0.6, ax=ax)
    
    # Adicionando título e rótulos aos eixos
    ax.set_title('Relação entre KM driven e Preço', fontsize=16)
    ax.set_xlabel('Quilometragem (KM)', fontsize=12)
    ax.set_ylabel('Preço (R$)', fontsize=12)
    
    # Exibindo o gráfico no Streamlit
    st.pyplot(fig)

# Função para gerar a matriz de correlação e heatmap
def plotar_matriz_correlacao(data):
    """
    Esta função gera a matriz de correlação entre todas as variáveis numéricas do dataset
    e exibe um heatmap para facilitar a visualização das correlações.
    """
    st.write("### Matriz de Correlação entre Variáveis")

    # Calculando a matriz de correlação entre todas as colunas numéricas
    corr = data.corr()

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

        # Visualizar a distribuição de preços
        plotar_distribuicao_precos(data)
        
        # Visualizar a relação entre KM driven e Preço
        plotar_relacao_km_preco(data)
        
        # Visualizar a matriz de correlação
        plotar_matriz_correlacao(data)
    else:
        st.error("Dados não carregados. Verifique a disponibilidade do dataset.")

# Executando a função principal
if __name__ == "__main__":
    main()

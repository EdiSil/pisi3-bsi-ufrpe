import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Função para carregar o arquivo CSV
def carregar_arquivo():
    arquivo = st.file_uploader("Carregue o arquivo CSV (OLX_cars_novo.csv)", type=["csv"])
    if arquivo is not None:
        # Lê o arquivo CSV e retorna um DataFrame
        return pd.read_csv(arquivo)
    return None

# Função para aplicar K-Means clustering
def aplicar_kmeans(data, n_clusters):
    # Preparar os dados para clusterização
    features = data.select_dtypes(include=['float64', 'int64'])  # Seleciona apenas colunas numéricas
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Aplicar K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    # Adicionar os rótulos de cluster ao dataset
    data['Cluster'] = clusters

    return data, clusters, kmeans

# Função para exibir os resultados de análise exploratória
def exibir_resultados(data, clusters, kmeans):
    st.write("Resumo dos Clusters")
    
    # Resumo dos clusters
    cluster_summary = data.groupby('Cluster').mean()
    st.write(cluster_summary)

    # Exibir gráfico de dispersão dos clusters
    st.write("Gráfico de Dispersão dos Clusters")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue=data['Cluster'], palette='viridis')
    plt.title('Distribuição dos Clusters')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    st.pyplot()

    # Elbow Method para determinar o número ideal de clusters
    st.write("Método do Cotovelo para Determinação do Número de Clusters")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(data.select_dtypes(include=['float64', 'int64']))
        wcss.append(kmeans.inertia_)
    
    # Gráfico do Método do Cotovelo
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), wcss)
    plt.title('Método do Cotovelo')
    plt.xlabel('Número de Clusters')
    plt.ylabel('WCSS')
    st.pyplot()

    # Exibir os primeiros rótulos de cluster
    st.write("Primeiros 10 rótulos de cluster:")
    st.write(clusters[:10])

# Função principal do Streamlit
def main():
    st.title("Análise de Clustering com K-Means")

    # Carregar o dataset
    data = carregar_arquivo()
    
    if data is not None:
        # Exibir os dados carregados
        st.write("Dados Carregados:")
        st.write(data.head())

        # Perguntar ao usuário o número de clusters desejado
        n_clusters = st.slider("Selecione o número de clusters", min_value=2, max_value=10, value=3)

        # Aplicar K-Means clustering
        dados_com_clusters, clusters, kmeans = aplicar_kmeans(data, n_clusters)

        # Exibir os resultados da análise exploratória
        exibir_resultados(dados_com_clusters, clusters, kmeans)

# Executar o Streamlit
if __name__ == "__main__":
    main()

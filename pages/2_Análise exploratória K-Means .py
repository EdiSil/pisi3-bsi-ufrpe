import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# URL do arquivo CSV no GitHub (link RAW)
URL_CSV = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_novo.csv"

# Função para carregar o arquivo CSV
def carregar_arquivo():
    try:
        # Carregar o arquivo CSV diretamente da URL
        data = pd.read_csv(URL_CSV)
        return data
    except Exception as e:
        st.error("Erro ao carregar o arquivo. Verifique o link ou o formato do arquivo.")
        st.write(e)
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
    
    # Garantir que apenas as colunas numéricas sejam usadas para calcular a média
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    cluster_summary = data.groupby('Cluster')[numeric_columns].mean()
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

# Função para salvar o DataFrame limpo e permitir o download
def salvar_arquivo(data):
    # Salvar o DataFrame como um arquivo CSV localmente
    data.to_csv('OLX_cars_com_clusters.csv', index=False)
    st.success("Arquivo com clusters salvo com sucesso como 'OLX_cars_com_clusters.csv'.")
    
    # Adicionar botão de download para o arquivo limpo
    st.download_button(
        label="Baixar arquivo CSV com clusters",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name='OLX_cars_com_clusters.csv',
        mime='text/csv'
    )

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

        # Salvar e permitir download do arquivo
        salvar_arquivo(dados_com_clusters)

# Executar o Streamlit
if __name__ == "__main__":
    main()

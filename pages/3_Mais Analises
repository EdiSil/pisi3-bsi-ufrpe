import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Classe para Análise de Clusters de Carros
class CarClusterAnalysis:
    def __init__(self, data):
        self.data = data
        self.features = ['ano', 'full_range', 'quilometragem', 'preco', 'Car Age']
        self.X = self.data[self.features]
        self.kmeans = None

    def elbow_method(self, max_clusters=10):
        inertia = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X)
            inertia.append(kmeans.inertia_)

        # Plotando o método do cotovelo
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=range(1, max_clusters + 1), y=inertia, marker='o', ax=ax)
        ax.set_title('Método do Cotovelo')
        ax.set_xlabel('Número de Clusters')
        ax.set_ylabel('Inércia')
        ax.grid(True)
        st.pyplot(fig)

    def perform_clustering(self, n_clusters):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.data['Cluster_Pred'] = self.kmeans.fit_predict(self.X)

        # Gerar uma paleta de cores diferentes para os clusters
        palette = sns.color_palette("Set2", n_colors=n_clusters)

        # Criando o gráfico de dispersão
        fig, ax = plt.subplots(figsize=(10, 6))

        for cluster_num in range(n_clusters):
            cluster_data = self.data[self.data['Cluster_Pred'] == cluster_num]
            sns.scatterplot(
                x=cluster_data['quilometragem'], 
                y=cluster_data['preco'], 
                color=palette[cluster_num],
                label=f'Cluster {cluster_num} - {len(cluster_data)} carros', 
                s=60,
                ax=ax
            )

        ax.set_title(f'Clusters de Carros - {n_clusters} Clusters')
        ax.set_xlabel('Quilometragem')
        ax.set_ylabel('Preço')
        ax.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=10)
        plt.tight_layout()
        ax.grid(True)
        st.pyplot(fig)

    def plot_confusion_matrix(self):
        y_true = self.data['Cluster']  # Certifique-se de que a coluna 'Cluster' existe
        y_pred = self.data['Cluster_Pred']

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='magma', cbar=True, linewidths=1, linecolor='black', ax=ax)
        ax.set_title('Matriz de Confusão Normalizada - Clusterização')
        ax.set_ylabel('Clusters Reais')
        ax.set_xlabel('Clusters Preditos')
        plt.ylim(0, cm.shape[0])
        st.pyplot(fig)

# Configurando a interface do Streamlit
def main():
    st.title("Análise de Clusters de Carros")
    st.sidebar.title("Configurações")

    # Carregar o arquivo de dados
    file_path = st.sidebar.file_uploader("Carregue o arquivo CSV dos carros", type=["csv"])
    if file_path:
        df = pd.read_csv(file_path)
        st.write("Prévia dos dados carregados:")
        st.dataframe(df.head())

        # Criar instância da classe
        analysis = CarClusterAnalysis(df)

        # Executar o método do cotovelo
        st.header("Método do Cotovelo")
        max_clusters = st.sidebar.slider("Número máximo de clusters", min_value=2, max_value=20, value=10, step=1)
        analysis.elbow_method(max_clusters=max_clusters)

        # Seleção do número de clusters e clustering
        st.header("Clusterização")
        n_clusters = st.sidebar.slider("Número de clusters", min_value=2, max_value=20, value=5, step=1)
        if st.sidebar.button("Executar Clusterização"):
            analysis.perform_clustering(n_clusters=n_clusters)
            st.write("Clusterização concluída!")

        # Matriz de Confusão
        if 'Cluster' in df.columns:
            st.header("Matriz de Confusão")
            analysis.plot_confusion_matrix()

# Executar a aplicação
if __name__ == "__main__":
    main()

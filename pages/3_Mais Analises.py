import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Classe para Análise de Cluster de Carros
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
        if 'Cluster' not in self.data.columns:
            st.warning("A coluna 'Cluster' com os clusters reais não está disponível no dataset.")
            return

        y_true = self.data['Cluster']
        y_pred = self.data['Cluster_Pred']

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Ajustando a escala de acordo com os valores reais da matriz de confusão
        vmin = np.min(cm_norm)
        vmax = np.max(cm_norm)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='magma', cbar=True, linewidths=1, linecolor='black', vmin=vmin, vmax=vmax, ax=ax)
        ax.set_title('Matriz de Confusão Normalizada - Clusterização')
        ax.set_ylabel('Clusters Reais')
        ax.set_xlabel('Clusters Preditos')
        plt.ylim(0, cm.shape[0])
        st.pyplot(fig)

# Interface no Streamlit
def main():
    st.title("Análise de Clusters de Carros")
    st.markdown("Explore os clusters de carros de forma interativa.")

    # Upload do arquivo
    file_path = st.file_uploader("Carregue o arquivo CSV:", type="csv")

    if file_path is not None:
        # Carregando os dados
        df = pd.read_csv(file_path)
        st.success("Dados carregados com sucesso!")
        
        # Exibindo um resumo dos dados
        st.write("Visualizando as primeiras linhas do dataset:")
        st.dataframe(df.head())

        # Criando uma instância da classe
        analysis = CarClusterAnalysis(df)

        # Painel de controle para o usuário
        st.sidebar.header("Configurações")
        max_clusters = st.sidebar.slider("Número máximo de clusters (Elbow Method):", 2, 15, 10)
        num_clusters = st.sidebar.slider("Número de clusters (Análise):", 2, 10, 5)

        # Exibir os gráficos
        st.subheader("Método do Cotovelo")
        analysis.elbow_method(max_clusters=max_clusters)

        st.subheader(f"Clusters com {num_clusters} Grupos")
        analysis.perform_clustering(n_clusters=num_clusters)

        st.subheader("Matriz de Confusão Normalizada")
        analysis.plot_confusion_matrix()

if __name__ == "__main__":
    main()


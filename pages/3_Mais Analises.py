import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

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
        ax.set_title('MÉTODO DO COTOVELO')
        ax.set_xlabel('NÚMERO DE CLUSTERS')
        ax.set_ylabel('INÉRCIA')
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
                label=f'CLUSTER {cluster_num} - {len(cluster_data)} CARROS',
                s=60,
                ax=ax
            )

        ax.set_title(f'CLUSTERS DE CARROS - {n_clusters} CLUSTERS')
        ax.set_xlabel('QUILOMETRAGEM')
        ax.set_ylabel('PREÇO')
        ax.legend(title='CLUSTERS', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=10)
        plt.tight_layout()
        ax.grid(True)
        st.pyplot(fig)

    def plot_confusion_matrix(self):
        if 'Cluster' not in self.data.columns:
            st.warning("A COLUNA 'CLUSTER' COM OS CLUSTERS REAIS NÃO ESTÁ DISPONÍVEL NO DATASET.")
            return

        y_true = self.data['Cluster']
        y_pred = self.data['Cluster_Pred']

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Garantindo que o valor mínimo não seja zero se a matriz for muito homogênea
        vmin = np.min(cm_norm)
        vmax = np.max(cm_norm)
        
        # Evitar que a matriz tenha um intervalo muito pequeno
        if vmax - vmin < 0.01:
            vmax = vmin + 0.01

        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='magma', cbar=True, linewidths=1, 
                              linecolor='black', vmin=vmin, vmax=vmax, ax=ax)

        # Ajustando a legenda de cores
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Proporção', rotation=270, labelpad=20)

        # Exibir todos os valores possíveis na escala de cores
        ticks = np.linspace(vmin, vmax, num=5)  # Garantir que os ticks estejam corretamente espaçados
        cbar.set_ticks(ticks)  # Atualizar a escala de cores de acordo com os valores reais
        cbar.ax.tick_params(labelsize=10)

        ax.set_title('MATRIZ DE CONFUSÃO NORMALIZADA - CLUSTERIZAÇÃO')
        ax.set_ylabel('CLUSTERS REAIS')
        ax.set_xlabel('CLUSTERS PREDITOS')
        plt.ylim(0, cm.shape[0])
        st.pyplot(fig)

# Interface no Streamlit
def main():
    st.title("ANÁLISE DE CLUSTERS DE CARROS")
    st.markdown("EXPLORE OS CLUSTERS DE CARROS DE FORMA INTERATIVA.")

    # Caminho fixo para o arquivo
    file_path = 'Datas/2_Cars_clusterizado.csv'

    if os.path.exists(file_path):
        # Carregando os dados
        df = pd.read_csv(file_path)
        st.success("DADOS CARREGADOS COM SUCESSO!")

        # Criando uma instância da classe
        analysis = CarClusterAnalysis(df)

        # Painel de controle para o usuário
        st.sidebar.header("CONFIGURAÇÕES")
        max_clusters = st.sidebar.slider("NÚMERO MÁXIMO DE CLUSTERS (MÉTODO DO COTOVELO):", 2, 12, 12)
        num_clusters = st.sidebar.slider("NÚMERO DE CLUSTERS (MATRIZ):", 1, 5, 5)

        # Exibir os gráficos
        st.subheader("MÉTODO DO COTOVELO")
        analysis.elbow_method(max_clusters=max_clusters)

        st.subheader(f"CLUSTERS COM {num_clusters} GRUPOS")
        analysis.perform_clustering(n_clusters=num_clusters)

        st.subheader("MATRIZ DE CONFUSÃO NORMALIZADA")
        analysis.plot_confusion_matrix()

    else:
        st.error(f"ARQUIVO NÃO ENCONTRADO NO CAMINHO: {file_path}")

if __name__ == "__main__":
    main()


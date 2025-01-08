import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import confusion_matrix  # Adicionando a importação
import numpy as np  # Adicionando a importação de numpy

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
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(1, max_clusters + 1), y=inertia, marker='o')
        plt.title('Método do Cotovelo')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Inércia')
        plt.grid(True)
        plt.show()

    def perform_clustering(self, n_clusters):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.data['Cluster_Pred'] = self.kmeans.fit_predict(self.X)

        # Gerar uma paleta de cores diferentes para os clusters
        palette = sns.color_palette("Set2", n_colors=n_clusters)
        
        # Criando o gráfico de dispersão
        plt.figure(figsize=(10, 6))

        # Plotando cada cluster com cores diferentes
        for cluster_num in range(n_clusters):
            cluster_data = self.data[self.data['Cluster_Pred'] == cluster_num]
            sns.scatterplot(x=cluster_data['quilometragem'], 
                            y=cluster_data['preco'], 
                            color=palette[cluster_num],  # Usando 'color' no lugar de 'c'
                            label=f'Cluster {cluster_num} - {len(cluster_data)} carros', 
                            s=60)

        # Exibindo título e rótulos
        plt.title(f'Clusters de Carros - {n_clusters} Clusters')
        plt.xlabel('Quilometragem')
        plt.ylabel('Preço')
        
        # Melhorando a legenda
        plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=10)
        
        # Ajustando o layout para que a legenda não sobreponha o gráfico
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        
    def plot_confusion_matrix(self):
        y_true = self.data['Cluster']
        y_pred = self.data['Cluster_Pred']
        
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Ajustando a escala de acordo com os valores reais da matriz de confusão
        vmin = np.min(cm_norm)
        vmax = np.max(cm_norm)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='magma', cbar=True, linewidths=1, linecolor='black', vmin=vmin, vmax=vmax)
        plt.title('Matriz de Confusão Normalizada - Clusterização')
        plt.ylabel('Clusters Reais')
        plt.xlabel('Clusters Preditos')
        plt.ylim(0, cm.shape[0])
        plt.show()

if __name__ == "__main__":
    file_path = 'C:/Users/Tutu/Desktop/2_Cars_clusterizado.csv'
    df = pd.read_csv(file_path)
    analysis = CarClusterAnalysis(df)
    analysis.elbow_method(max_clusters=12)
    analysis.perform_clustering(n_clusters=5)  # Ajuste de acordo com o resultado do cotovelo
    analysis.plot_confusion_matrix()

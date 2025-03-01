import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.ticker as mticker
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configurações de estilo profissional
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    plt.style.use('ggplot')

sns.set_palette("husl")

class CarClusterAnalysis:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.kmeans = None
        self.n_clusters = None
        self.features = None
        self.feature_map = {
            'QUILOMETRAGEM': 'quilometragem',
            'PREÇO': 'preco',
            'ANO': 'ano',
            'AUTONOMIA': 'full_range',
            'IDADE DO VEÍCULO': 'Car Age'
        }

    def prepare_data(self, features):
        """Prepara os dados para análise com normalização"""
        try:
            self.features = [self.feature_map[f] for f in features]
            X = self.data[self.features]
            return self.scaler.fit_transform(X)
        except KeyError as e:
            st.error(f"ERRO: VARIÁVEL {e} NÃO ENCONTRADA NO DATASET")
            return None

    def calculate_elbow(self, X, max_clusters=15):
        """Calcula a curva do cotovelo para seleção de clusters"""
        inertia = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
        return inertia

    def calculate_silhouette(self, X, max_clusters=15):
        """Calcula os scores de silhueta"""
        scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(X)
            if len(np.unique(labels)) > 1:
                scores.append(silhouette_score(X, labels))
            else:
                scores.append(0)
        return scores

    def perform_clustering(self, X, n_clusters):
        """Executa o algoritmo K-Means"""
        try:
            if n_clusters < 2:
                raise ValueError("NÚMERO DE CLUSTERS DEVE SER PELO MENOS 2")
                
            self.n_clusters = n_clusters
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            return self.kmeans.fit_predict(X)
        except Exception as e:
            st.error(f"ERRO NO CLUSTERING: {str(e)}")
            return None

    def plot_confusion_matrix(self, true_labels, pred_labels):
        """Gera a matriz de confusão"""
        try:
            cm = confusion_matrix(true_labels, pred_labels)
            plt.figure(figsize=(12, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title('MATRIZ DE CONFUSÃO', fontweight='bold', fontsize=16, pad=20)
            plt.xlabel('CLUSTERS PREVISTOS', fontweight='bold')
            plt.ylabel('CLUSTERS REAIS', fontweight='bold')
            st.pyplot(plt)
        except Exception as e:
            st.error(f"ERRO NA MATRIZ DE CONFUSÃO: {str(e)}")

    def reduce_dimensionality(self, X):
        """Reduz a dimensionalidade para visualização 2D"""
        pca = PCA(n_components=2)
        return pca.fit_transform(X)

class ClusterVisualizer:
    def __init__(self):
        self.label_style = {'fontweight': 'bold', 'fontsize': 12}

    def format_number(self, x, pos):
        return f'{x:,.0f}'.replace(",", ".")
    
    def format_currency(self, x, pos):
        return f'US$ {x:,.0f}'.replace(",", ".")

    def plot_elbow(self, inertia, max_clusters):
        """Visualiza o método do cotovelo"""
        plt.figure(figsize=(14, 7))
        sns.lineplot(x=range(1, max_clusters+1), y=inertia, marker='o', linewidth=2)
        plt.title('MÉTODO DO COTOVELO', fontweight='bold', fontsize=18, pad=20)
        plt.xlabel('NÚMERO DE CLUSTERS', **self.label_style)
        plt.ylabel('INÉRCIA', **self.label_style)
        plt.grid(True, alpha=0.3)
        st.pyplot(plt)

    def plot_silhouette_scores(self, scores, max_clusters):
        """Visualiza os scores médios de silhueta"""
        plt.figure(figsize=(14, 7))
        sns.lineplot(x=range(2, max_clusters+1), y=scores, marker='o', linewidth=2)
        plt.title('SCORE MÉDIO DE SILHUETA', fontweight='bold', fontsize=18, pad=20)
        plt.xlabel('NÚMERO DE CLUSTERS', **self.label_style)
        plt.ylabel('SCORE DE SILHUETA', **self.label_style)
        plt.grid(True, alpha=0.3)
        st.pyplot(plt)

    def plot_silhouette_analysis(self, X, labels, n_clusters):
        """Análise detalhada de silhueta"""
        try:
            plt.figure(figsize=(14, 10))
            silhouette_avg = silhouette_score(X, labels)
            sample_silhouette_values = silhouette_samples(X, labels)

            y_lower = 10
            colors = sns.husl_palette(n_clusters)
            
            for i in range(n_clusters):
                cluster_values = sample_silhouette_values[labels == i]
                cluster_values.sort()

                y_upper = y_lower + cluster_values.size
                plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values,
                                facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
                plt.text(-0.05, y_lower + 0.5 * cluster_values.size, str(i),
                        fontweight='bold', fontsize=12)
                y_lower = y_upper + 10

            plt.title('ANÁLISE DE SILHUETA POR CLUSTER', fontweight='bold', fontsize=18, pad=20)
            plt.xlabel('COEFICIENTE DE SILHUETA', **self.label_style)
            plt.ylabel('CLUSTER', **self.label_style)
            plt.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2)
            plt.yticks([])
            plt.grid(True, alpha=0.3)
            st.pyplot(plt)
        except Exception as e:
            st.error(f"ERRO NA ANÁLISE DE SILHUETA: {str(e)}")

    def plot_cluster_distribution(self, data):
        """Visualiza a distribuição dos clusters"""
        plt.figure(figsize=(14, 7))
        cluster_dist = data['Cluster'].value_counts().sort_index()
        bars = sns.barplot(x=cluster_dist.index, y=cluster_dist.values, palette="husl")
        
        plt.title('DISTRIBUIÇÃO DOS CLUSTERS', fontweight='bold', fontsize=18, pad=20)
        plt.xlabel('CLUSTER', **self.label_style)
        plt.ylabel('QUANTIDADE', **self.label_style)
        
        for bar in bars.patches:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{int(bar.get_height())}\n({bar.get_height()/len(data)*100:.1f}%)',
                    ha='center', va='center', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        st.pyplot(plt)

    def plot_pca_clusters(self, X_pca, labels):
        """Visualização 2D dos clusters"""
        plt.figure(figsize=(14, 10))
        unique_labels = np.unique(labels)
        colors = sns.husl_palette(len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1],
                       color=colors[i], s=100, edgecolor='w', linewidth=0.5,
                       label=f'CLUSTER {label}')
            
        plt.title('VISUALIZAÇÃO DOS CLUSTERS', fontweight='bold', fontsize=18, pad=20)
        plt.xlabel('COMPONENTE PRINCIPAL 1', **self.label_style)
        plt.ylabel('COMPONENTE PRINCIPAL 2', **self.label_style)
        plt.legend(title='CLUSTER', bbox_to_anchor=(1.05, 1))
        plt.grid(True, alpha=0.3)
        st.pyplot(plt)

def main():
    st.set_page_config(page_title="Análise de Clusters", layout="wide")
    st.title("🚗 ANÁLISE DE CLUSTERS DE VEÍCULOS")
    
    # Carregamento de dados
    file_path = 'Datas/2_Cars_clusterizado.csv'
    if not os.path.exists(file_path):
        st.error("ERRO: ARQUIVO NÃO ENCONTRADO!")
        return

    try:
        df = pd.read_csv(file_path)
        analyzer = CarClusterAnalysis(df)
        visualizer = ClusterVisualizer()
    except Exception as e:
        st.error(f"ERRO AO CARREGAR DADOS: {str(e)}")
        return

    # Sidebar
    st.sidebar.header("CONFIGURAÇÕES")
    selected_features = st.sidebar.multiselect(
        "SELECIONE AS VARIÁVEIS PARA CLUSTERING:",
        options=['QUILOMETRAGEM', 'PREÇO', 'ANO', 'AUTONOMIA', 'IDADE DO VEÍCULO'],
        default=['QUILOMETRAGEM', 'PREÇO', 'ANO']
    )
    
    n_clusters = st.sidebar.slider(
        "NÚMERO DE CLUSTERS PARA ANÁLISE:",
        2, 15, 5
    )

    # Processamento principal
    try:
        X = analyzer.prepare_data(selected_features)
        if X is None:
            return

        # Análises preliminares
        st.header("ANÁLISE DE CLUSTERS")
        
        st.subheader("MÉTODO DO COTOVELO")
        inertia = analyzer.calculate_elbow(X, 15)
        visualizer.plot_elbow(inertia, 15)
        
        st.subheader("SCORE DE SILHUETA")
        scores = analyzer.calculate_silhouette(X, 15)
        visualizer.plot_silhouette_scores(scores, 15)

        # Clusterização
        labels = analyzer.perform_clustering(X, n_clusters)
        if labels is None:
            return
        
        df['Cluster_Pred'] = labels

        # Visualizações pós-clusterização
        st.subheader("ANÁLISE DE SILHUETA DETALHADA")
        visualizer.plot_silhouette_analysis(X, labels, n_clusters)
        
        st.subheader("DISTRIBUIÇÃO DOS CLUSTERS")
        visualizer.plot_cluster_distribution(df)
        
        st.subheader("VISUALIZAÇÃO DOS CLUSTERS")
        X_pca = analyzer.reduce_dimensionality(X)
        visualizer.plot_pca_clusters(X_pca, labels)
        
        st.subheader("MATRIZ DE CONFUSÃO")
        analyzer.plot_confusion_matrix(df['Cluster'], df['Cluster_Pred'])

    except Exception as e:
        st.error(f"ERRO: {str(e)}")

if __name__ == "__main__":
    main()

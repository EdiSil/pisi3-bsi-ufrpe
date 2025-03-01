import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.ticker as mticker
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
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
        self.feature_names = {
            'quilometragem': 'QUILOMETRAGEM (KM)',
            'preco': 'PREÇO (USD)',
            'ano': 'ANO DE FABRICAÇÃO',
            'full_range': 'AUTONOMIA (KM)',
            'Car Age': 'IDADE DO VEÍCULO (ANOS)'
        }

    def prepare_data(self, features):
        """Prepara os dados para análise com normalização"""
        try:
            self.features = features
            X = self.data[features]
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
        """Calcula os scores de silhueta para diferentes números de clusters"""
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
        """Executa o algoritmo K-Means e retorna os labels"""
        try:
            self.n_clusters = n_clusters
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            return self.kmeans.fit_predict(X)
        except Exception as e:
            st.error(f"ERRO NO CLUSTERING: {str(e)}")
            return None

    def reduce_dimensionality(self, X):
        """Reduz a dimensionalidade para visualização 2D"""
        pca = PCA(n_components=2)
        return pca.fit_transform(X)

class ClusterVisualizer:
    def __init__(self):
        self.format_config = {
            'quilometragem': ('QUILOMETRAGEM (KM)', self.format_number),
            'preco': ('PREÇO (USD)', self.format_currency),
            'ano': ('ANO DE FABRICAÇÃO', self.format_number),
            'full_range': ('AUTONOMIA (KM)', self.format_number),
            'Car Age': ('IDADE DO VEÍCULO (ANOS)', self.format_number)
        }

    def format_number(self, x, pos):
        return f'{x:,.0f}'.replace(",", ".")
    
    def format_currency(self, x, pos):
        return f'US$ {x:,.0f}'.replace(",", ".")

    def plot_elbow(self, inertia, max_clusters):
        """Visualiza o método do cotovelo"""
        plt.figure(figsize=(14, 7))
        ax = sns.lineplot(x=range(1, max_clusters+1), y=inertia, marker='o', linewidth=2)
        plt.title('MÉTODO DO COTOVELO', fontweight='bold', fontsize=18, pad=20)
        plt.xlabel('NÚMERO DE CLUSTERS', fontweight='bold', labelpad=15)
        plt.ylabel('INÉRCIA', fontweight='bold', labelpad=15)
        plt.grid(True, alpha=0.3)
        st.pyplot(plt)

    def plot_silhouette(self, scores, max_clusters):
        """Visualiza os scores de silhueta"""
        plt.figure(figsize=(14, 7))
        ax = sns.lineplot(x=range(2, max_clusters+1), y=scores, marker='o', linewidth=2)
        plt.title('ANÁLISE DE SILHUETA', fontweight='bold', fontsize=18, pad=20)
        plt.xlabel('NÚMERO DE CLUSTERS', fontweight='bold', labelpad=15)
        plt.ylabel('SCORE MÉDIO', fontweight='bold', labelpad=15)
        plt.grid(True, alpha=0.3)
        st.pyplot(plt)

    def plot_cluster_distribution(self, data):
        """Visualiza a distribuição dos clusters"""
        plt.figure(figsize=(14, 7))
        cluster_dist = data['Cluster'].value_counts().sort_index()
        bars = sns.barplot(x=cluster_dist.index, y=cluster_dist.values, palette="husl")
        
        plt.title('DISTRIBUIÇÃO DOS CLUSTERS', fontweight='bold', fontsize=18, pad=20)
        plt.xlabel('CLUSTER', fontweight='bold', labelpad=15)
        plt.ylabel('QUANTIDADE DE VEÍCULOS', fontweight='bold', labelpad=15)
        
        for bar in bars.patches:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{int(bar.get_height())}\n({bar.get_height()/len(data)*100:.1f}%)',
                    ha='center', va='center', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        st.pyplot(plt)

    def plot_pca_clusters(self, X_pca, labels):
        """Visualiza clusters em espaço 2D reduzido"""
        plt.figure(figsize=(14, 10))
        scatter = sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels,
                                 palette="husl", s=100, edgecolor='w', linewidth=0.5)
        
        plt.title('VISUALIZAÇÃO DE CLUSTERS EM 2D', fontweight='bold', fontsize=18, pad=20)
        plt.xlabel('COMPONENTE PRINCIPAL 1', fontweight='bold', labelpad=15)
        plt.ylabel('COMPONENTE PRINCIPAL 2', fontweight='bold', labelpad=15)
        plt.legend(title='CLUSTER', bbox_to_anchor=(1.05, 1), borderaxespad=0)
        plt.grid(True, alpha=0.3)
        st.pyplot(plt)

    def plot_parallel_coordinates(self, data, features, labels):
        """Visualização multivariada por coordenadas paralelas"""
        plt.figure(figsize=(16, 10))
        numeric_data = data[features].apply(pd.to_numeric, errors='coerce')
        numeric_data['Cluster'] = labels
        
        plt.title('PERFIL MULTIVARIADO DOS CLUSTERS', fontweight='bold', fontsize=18, pad=20)
        pd.plotting.parallel_coordinates(numeric_data, 'Cluster', color=("#FF0000", "#00FF00", "#0000FF", "#FF00FF", "#00FFFF"))
        plt.xlabel('CARACTERÍSTICAS', fontweight='bold', labelpad=15)
        plt.ylabel('VALORES NORMALIZADOS', fontweight='bold', labelpad=15)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(plt)

def main():
    st.set_page_config(page_title="Análise de Clusters de Veículos", layout="wide")
    st.title("🚗 ANÁLISE AVANÇADA DE CLUSTERS DE VEÍCULOS")
    
    # Carregamento de dados
    file_path = 'Datas/2_Cars_clusterizado.csv'
    if not os.path.exists(file_path):
        st.error("ERRO: ARQUIVO DE DADOS NÃO ENCONTRADO!")
        return

    try:
        df = pd.read_csv(file_path)
        analyzer = CarClusterAnalysis(df)
        visualizer = ClusterVisualizer()
    except Exception as e:
        st.error(f"ERRO NA CARGA DE DADOS: {str(e)}")
        return

    # Configurações da sidebar
    st.sidebar.header("CONFIGURAÇÕES DA ANÁLISE")
    
    feature_map = {
        'QUILOMETRAGEM': 'quilometragem',
        'PREÇO': 'preco',
        'ANO': 'ano',
        'AUTONOMIA': 'full_range',
        'IDADE DO VEÍCULO': 'Car Age'
    }
    
    selected_features = st.sidebar.multiselect(
        "SELECIONE AS VARIÁVEIS PARA ANÁLISE:",
        options=list(feature_map.keys()),
        default=['QUILOMETRAGEM', 'PREÇO', 'ANO']
    )
    
    feature_keys = [feature_map[f] for f in selected_features]
    
    n_clusters = st.sidebar.slider(
        "NÚMERO DE CLUSTERS PARA MODELAGEM:",
        2, 15, 5
    )

    # Processamento principal
    try:
        X = analyzer.prepare_data(feature_keys)
        if X is None:
            return

        st.header("ANÁLISE EXPLORATÓRIA DE CLUSTERS")
        
        # Seção de Métricas de Clusterização
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("MÉTODO DO COTOVELO")
            inertia = analyzer.calculate_elbow(X, 15)
            visualizer.plot_elbow(inertia, 15)
        
        with col2:
            st.subheader("ANÁLISE DE SILHUETA")
            scores = analyzer.calculate_silhouette(X, 15)
            visualizer.plot_silhouette(scores, 15)

        # Clusterização e Visualizações
        st.subheader(f"MODELAGEM COM {n_clusters} CLUSTERS")
        labels = analyzer.perform_clustering(X, n_clusters)
        if labels is None:
            return
        
        df['Cluster'] = labels
        X_pca = analyzer.reduce_dimensionality(X)

        # Visualizações dos Resultados
        st.subheader("DISTRIBUIÇÃO DOS CLUSTERS")
        visualizer.plot_cluster_distribution(df)
        
        st.subheader("VISUALIZAÇÃO MULTIDIMENSIONAL")
        visualizer.plot_parallel_coordinates(df, feature_keys, labels)
        
        st.subheader("PROJEÇÃO EM 2D")
        visualizer.plot_pca_clusters(X_pca, labels)

    except Exception as e:
        st.error(f"ERRO NO PROCESSAMENTO: {str(e)}")

if __name__ == "__main__":
    main()

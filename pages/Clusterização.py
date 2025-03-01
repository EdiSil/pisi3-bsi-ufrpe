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

# Configuração de estilo profissional
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
            'quilometragem': 'QUILOMETRAGEM',
            'preco': 'PREÇO',
            'ano': 'ANO',
            'full_range': 'AUTONOMIA',
            'Car Age': 'IDADE DO VEÍCULO'
        }

    def prepare_data(self, features):
        """Prepara os dados para clustering com normalização"""
        try:
            self.features = features
            X = self.data[features]
            return self.scaler.fit_transform(X)
        except KeyError as e:
            st.error(f"Variável não encontrada: {e}")
            return None

    def calculate_elbow(self, X, max_clusters=15):
        """Calcula os valores de inércia para o método do cotovelo"""
        inertia = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
        return inertia

    def calculate_silhouette(self, X, max_clusters=15):
        """Calcula os scores de silhueta"""
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(X)
            if len(np.unique(labels)) > 1:
                silhouette_scores.append(silhouette_score(X, labels))
            else:
                silhouette_scores.append(0)
        return silhouette_scores

    def perform_clustering(self, X, n_clusters):
        """Executa o clustering K-means"""
        try:
            self.n_clusters = n_clusters
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            return self.kmeans.fit_predict(X)
        except Exception as e:
            st.error(f"Erro no clustering: {e}")
            return None

    def plot_confusion_matrix(self, labels):
        """Plota a matriz de confusão"""
        try:
            if 'Cluster' not in self.data.columns:
                st.warning("Dados originais de cluster não encontrados")
                return

            cm = confusion_matrix(self.data['Cluster'], labels)
            plt.figure(figsize=(14, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('MATRIZ DE CONFUSÃO - CLUSTERS', fontweight='bold', pad=20)
            plt.xlabel('Clusters Previstos', fontweight='bold')
            plt.ylabel('Clusters Reais', fontweight='bold')
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Erro na matriz de confusão: {e}")

class ClusterVisualizer:
    def __init__(self):
        self.format_config = {
            'quilometragem': ('QUILOMETRAGEM (Km)', self.format_number),
            'preco': ('PREÇO (USD)', self.format_currency),
            'ano': ('ANO', self.format_number),
            'full_range': ('AUTONOMIA (Km)', self.format_number),
            'Car Age': ('IDADE DO VEÍCULO (Anos)', self.format_number)
        }

    def format_number(self, x, pos):
        return f'{x:,.0f}'.replace(",", ".")
    
    def format_currency(self, x, pos):
        return f'US$ {x:,.0f}'.replace(",", ".")

    def plot_elbow(self, inertia, max_clusters):
        """Plota o gráfico do método do cotovelo"""
        plt.figure(figsize=(14, 7))
        sns.lineplot(x=range(1, max_clusters+1), y=inertia, marker='o', linewidth=2)
        plt.title('MÉTODO DO COTOVELO', fontweight='bold', fontsize=16, pad=20)
        plt.xlabel('Número de Clusters', fontweight='bold')
        plt.ylabel('Inércia', fontweight='bold')
        plt.grid(True, alpha=0.5)
        st.pyplot(plt)

    def plot_silhouette(self, scores, max_clusters):
        """Plota os scores de silhueta"""
        plt.figure(figsize=(14, 7))
        sns.lineplot(x=range(2, max_clusters+1), y=scores, marker='o', linewidth=2)
        plt.title('SCORE DE SILHUETA', fontweight='bold', fontsize=16, pad=20)
        plt.xlabel('Número de Clusters', fontweight='bold')
        plt.ylabel('Score Médio', fontweight='bold')
        plt.grid(True, alpha=0.5)
        st.pyplot(plt)

    def plot_scatter(self, data, x_feat, y_feat, labels):
        """Plota gráfico de dispersão dos clusters"""
        plt.figure(figsize=(14, 8))
        scatter = sns.scatterplot(
            data=data,
            x=x_feat,
            y=y_feat,
            hue=labels,
            palette="husl",
            s=100,
            edgecolor='w',
            linewidth=0.5
        )
        
        # Formatação dos eixos
        x_label, x_formatter = self.format_config[x_feat]
        y_label, y_formatter = self.format_config[y_feat]
        
        plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(x_formatter))
        plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(y_formatter))
        
        plt.title(f'CLUSTERS: {x_label} vs {y_label}', fontweight='bold', fontsize=16, pad=20)
        plt.xlabel(x_label, fontweight='bold')
        plt.ylabel(y_label, fontweight='bold')
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), borderaxespad=0)
        plt.grid(True, alpha=0.3)
        st.pyplot(plt)

def main():
    st.set_page_config(page_title="Análise de Clusters", layout="wide")
    st.title("🚘 ANÁLISE DE CLUSTERS DE VEÍCULOS")

    # Carregamento de dados
    file_path = 'Datas/2_Cars_clusterizado.csv'
    if not os.path.exists(file_path):
        st.error("Arquivo não encontrado!")
        return

    try:
        df = pd.read_csv(file_path)
        analyzer = CarClusterAnalysis(df)
        visualizer = ClusterVisualizer()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return

    # Sidebar
    st.sidebar.header("CONFIGURAÇÕES")
    
    feature_map = {
        'QUILOMETRAGEM': 'quilometragem',
        'PREÇO': 'preco',
        'ANO': 'ano',
        'AUTONOMIA': 'full_range',
        'IDADE DO VEÍCULO': 'Car Age'
    }
    
    selected_features = st.sidebar.multiselect(
        "VARIÁVEIS PARA ANÁLISE:",
        options=list(feature_map.keys()),
        default=['QUILOMETRAGEM', 'PREÇO', 'ANO']
    )
    
    # Converter nomes para chaves
    feature_keys = [feature_map[f] for f in selected_features]
    
    n_clusters = st.sidebar.slider(
        "NÚMERO DE CLUSTERS PARA ANÁLISE:",
        2, 15, 5
    )

    # Análise principal
    try:
        X = analyzer.prepare_data(feature_keys)
        if X is None:
            return

        st.header("ANÁLISE DE CLUSTERIZAÇÃO")
        
        # Método do Cotovelo
        st.subheader("MÉTODO DO COTOVELO")
        inertia = analyzer.calculate_elbow(X, 15)
        visualizer.plot_elbow(inertia, 15)
        
        # Análise de Silhueta
        st.subheader("ANÁLISE DE SILHUETA")
        silhouette_scores = analyzer.calculate_silhouette(X, 15)
        visualizer.plot_silhouette(silhouette_scores, 15)
        
        # Clusterização
        st.subheader(f"CLUSTERIZAÇÃO COM {n_clusters} GRUPOS")
        labels = analyzer.perform_clustering(X, n_clusters)
        if labels is None:
            return
            
        # Gráfico de Dispersão
        st.subheader("DISTRIBUIÇÃO DOS CLUSTERS")
        visualizer.plot_scatter(df, feature_keys[0], feature_keys[1], labels)
        
        # Matriz de Confusão
        st.subheader("MATRIZ DE CONFUSÃO")
        analyzer.plot_confusion_matrix(labels)

    except Exception as e:
        st.error(f"Erro na análise: {e}")

if __name__ == "__main__":
    main()

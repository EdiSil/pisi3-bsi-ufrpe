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
            st.error(f"VARIÁVEL NÃO ENCONTRADA: {e}")
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
            st.error(f"ERRO NO CLUSTERING: {e}")
            return None

    def plot_confusion_matrix(self, labels):
        """Plota a matriz de confusão aprimorada"""
        try:
            if 'Cluster' not in self.data.columns:
                st.warning("DADOS ORIGINAIS DE CLUSTER NÃO ENCONTRADOS")
                return

            cm = confusion_matrix(self.data['Cluster'], labels)
            plt.figure(figsize=(14, 8))
            
            # Calcular porcentagens
            cm_sum = np.sum(cm, axis=1, keepdims=True)
            cm_percent = cm / cm_sum.astype(float) * 100
            
            # Plotar heatmap com valores absolutos e porcentagens
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        annot_kws={"size": 12}, cbar=False)
            
            # Adicionar porcentagens
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j+0.5, i+0.3, f"{cm_percent[i,j]:.1f}%",
                            ha='center', va='center', color='black', fontsize=10)
            
            plt.title('MATRIZ DE CONFUSÃO - DISTRIBUIÇÃO DE CLUSTERS', 
                     fontweight='bold', fontsize=16, pad=20)
            plt.xlabel('CLUSTERS PREVISTOS', fontweight='bold')
            plt.ylabel('CLUSTERS REAIS', fontweight='bold')
            st.pyplot(plt)
        except Exception as e:
            st.error(f"ERRO NA MATRIZ DE CONFUSÃO: {e}")

class ClusterVisualizer:
    def __init__(self):
        self.format_config = {
            'quilometragem': ('QUILOMETRAGEM (KM)', self.format_number),
            'preco': ('PREÇO (USD)', self.format_currency),
            'ano': ('ANO', self.format_number),
            'full_range': ('AUTONOMIA (KM)', self.format_number),
            'Car Age': ('IDADE DO VEÍCULO (ANOS)', self.format_number)
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
        plt.xlabel('NÚMERO DE CLUSTERS', fontweight='bold')
        plt.ylabel('INÉRCIA', fontweight='bold')
        plt.grid(True, alpha=0.5)
        st.pyplot(plt)

    def plot_silhouette(self, scores, max_clusters):
        """Plota os scores de silhueta"""
        plt.figure(figsize=(14, 7))
        sns.lineplot(x=range(2, max_clusters+1), y=scores, marker='o', linewidth=2)
        plt.title('SCORE DE SILHUETA', fontweight='bold', fontsize=16, pad=20)
        plt.xlabel('NÚMERO DE CLUSTERS', fontweight='bold')
        plt.ylabel('SCORE MÉDIO', fontweight='bold')
        plt.grid(True, alpha=0.5)
        st.pyplot(plt)

    def plot_cluster_distribution(self, data):
        """Plota a distribuição de clusters em barras sobrepostas"""
        try:
            plt.figure(figsize=(14, 8))
            cluster_dist = data['Cluster'].value_counts().sort_index()
            
            colors = sns.color_palette('husl', len(cluster_dist))
            
            bars = plt.bar(cluster_dist.index.astype(str), 
                          cluster_dist.values,
                          color=colors,
                          edgecolor='white',
                          linewidth=1)
            
            # Adicionar valores e porcentagens
            total = sum(cluster_dist)
            for bar in bars:
                height = bar.get_height()
                percent = (height / total) * 100
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height}\n({percent:.1f}%)',
                        ha='center', va='center',
                        fontweight='bold')
            
            plt.title('DISTRIBUIÇÃO DE CLUSTERS - CONTAGEM E PERCENTUAL', 
                     fontweight='bold', fontsize=16, pad=20)
            plt.xlabel('CLUSTERS', fontweight='bold')
            plt.ylabel('QUANTIDADE', fontweight='bold')
            plt.grid(True, axis='y', alpha=0.3)
            st.pyplot(plt)
        except Exception as e:
            st.error(f"ERRO AO PLOTAR DISTRIBUIÇÃO: {e}")

    def plot_parallel_coordinates(self, data, features, labels):
        """Plota coordenadas paralelas para visualização multivariada"""
        try:
            plt.figure(figsize=(16, 8))
            numeric_data = data[features].apply(pd.to_numeric, errors='coerce')
            numeric_data['Cluster'] = labels
            
            # Amostrar dados para melhor visualização
            if len(numeric_data) > 1000:
                sample_data = numeric_data.sample(1000)
            else:
                sample_data = numeric_data
            
            sns.lineplot(data=sample_data.melt(id_vars='Cluster'),
                        x='variable', 
                        y='value',
                        hue='Cluster',
                        palette='husl',
                        estimator='median',
                        errorbar=None)
            
            plt.title('PERFIL DOS CLUSTERS - COORDENADAS PARALELAS',
                    fontweight='bold', fontsize=16, pad=20)
            plt.xlabel('VARIÁVEIS', fontweight='bold')
            plt.ylabel('VALORES NORMALIZADOS', fontweight='bold')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), title='CLUSTER')
            st.pyplot(plt)
        except Exception as e:
            st.error(f"ERRO NA VISUALIZAÇÃO MULTIVARIADA: {e}")

def main():
    st.set_page_config(page_title="Análise de Clusters", layout="wide")
    st.title("🚘 ANÁLISE DE CLUSTERS DE VEÍCULOS")

    # Carregamento de dados
    file_path = 'Datas/2_Cars_clusterizado.csv'
    if not os.path.exists(file_path):
        st.error("ARQUIVO NÃO ENCONTRADO!")
        return

    try:
        df = pd.read_csv(file_path)
        analyzer = CarClusterAnalysis(df)
        visualizer = ClusterVisualizer()
    except Exception as e:
        st.error(f"ERRO AO CARREGAR DADOS: {e}")
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
        
        df['Cluster'] = labels
        
        # Gráfico de distribuição atualizado
        st.subheader("DISTRIBUIÇÃO DOS CLUSTERS")
        visualizer.plot_cluster_distribution(df)
        
        # Nova visualização multivariada
        st.subheader("PERFIL MULTIVARIADO DOS CLUSTERS")
        visualizer.plot_parallel_coordinates(df, feature_keys, labels)
        
        # Matriz de Confusão
        st.subheader("MATRIZ DE CONFUSÃO")
        analyzer.plot_confusion_matrix(labels)

    except Exception as e:
        st.error(f"ERRO NA ANÁLISE: {e}")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

class CarClusterAnalysis:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.kmeans = None
        self.n_clusters = None
        self.features = None

    def prepare_data(self, features):
        """Prepara os dados para clustering com normalização"""
        self.features = features
        X = self.data[features]
        return self.scaler.fit_transform(X)

    def calculate_elbow(self, X, max_clusters=10):
        """Calcula os valores de inércia para o método do cotovelo"""
        inertia = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
        return inertia

    def calculate_silhouette(self, X, max_clusters=10):
        """Calcula os scores de silhueta para diferentes números de clusters"""
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(X)
            silhouette_scores.append(silhouette_score(X, labels))
        return silhouette_scores

    def perform_clustering(self, X, n_clusters):
        """Executa o clustering K-means e retorna os labels"""
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        return self.kmeans.fit_predict(X)

    def plot_silhouette_analysis(self, X, labels):
        """Plota a análise de silhueta detalhada"""
        silhouette_avg = silhouette_score(X, labels)
        sample_silhouette_values = silhouette_samples(X, labels)

        fig, ax = plt.subplots(figsize=(10, 6))
        y_lower = 10
        
        for i in range(self.n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = sns.color_palette("husl", self.n_clusters)[i]
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax.set_title("Análise de Silhueta")
        ax.set_xlabel("Coeficiente de Silhueta")
        ax.set_ylabel("Cluster")
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax.set_yticks([])
        st.pyplot(fig)

class ClusterVisualizer:
    @staticmethod
    def plot_elbow(inertia, max_clusters):
        """Plota o gráfico do método do cotovelo"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=range(1, max_clusters + 1), y=inertia, marker='o', ax=ax)
        ax.set_title('Método do Cotovelo')
        ax.set_xlabel('Número de Clusters')
        ax.set_ylabel('Inércia')
        ax.grid(True)
        st.pyplot(fig)

    @staticmethod
    def plot_scatter(data, x_col, y_col, hue_col, palette):
        """Plota gráfico de dispersão com clusters"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, 
                       palette=palette, s=60, ax=ax)
        ax.set_title(f'Clusters: {x_col} vs {y_col}')
        ax.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        ax.grid(True)
        st.pyplot(fig)

    @staticmethod
    def plot_silhouette_scores(scores, max_clusters):
        """Plota os scores de silhueta médios"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=range(2, max_clusters + 1), y=scores, marker='o', ax=ax)
        ax.set_title('Pontuação Média de Silhueta')
        ax.set_xlabel('Número de Clusters')
        ax.set_ylabel('Score de Silhueta')
        ax.grid(True)
        st.pyplot(fig)

def main():
    st.set_page_config(page_title="Análise de Clusters de Carros", layout="wide")
    st.title("🚗 Análise de Clusters de Carros Interativa")
    
    # Carregamento de dados
    file_path = 'Datas/2_Cars_clusterizado.csv'
    
    if not os.path.exists(file_path):
        st.error(f"Arquivo não encontrado: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
        st.session_state['data'] = df
        st.success("Dados carregados com sucesso!")
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return

    # Sidebar controls
    st.sidebar.header("Configurações")
    available_features = ['ano', 'full_range', 'quilometragem', 'preco', 'Car Age']
    
    selected_features = st.sidebar.multiselect(
        "Selecione as variáveis para clustering:",
        options=available_features,
        default=['quilometragem', 'preco', 'ano']
    )
    
    max_clusters_elbow = st.sidebar.slider(
        "Número máximo de clusters (Cotovelo):",
        2, 15, 10
    )
    
    max_clusters_silhouette = st.sidebar.slider(
        "Número máximo de clusters (Silhueta):",
        2, 15, 8
    )
    
    n_clusters = st.sidebar.slider(
        "Número de clusters para análise detalhada:",
        2, 10, 5
    )
    
    # Inicialização do modelo
    analyzer = CarClusterAnalysis(df)
    visualizer = ClusterVisualizer()
    
    # Container principal
    with st.container():
        if len(selected_features) >= 2:
            try:
                X = analyzer.prepare_data(selected_features)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Método do Cotovelo")
                    inertia = analyzer.calculate_elbow(X, max_clusters_elbow)
                    visualizer.plot_elbow(inertia, max_clusters_elbow)
                
                with col2:
                    st.subheader("Análise de Silhueta")
                    silhouette_scores = analyzer.calculate_silhouette(X, max_clusters_silhouette)
                    visualizer.plot_silhouette_scores(silhouette_scores, max_clusters_silhouette)
                
                # Análise detalhada de clusters
                st.subheader(f"Análise Detalhada para {n_clusters} Clusters")
                labels = analyzer.perform_clustering(X, n_clusters)
                df['Cluster'] = labels
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.markdown("### Distribuição dos Clusters")
                    cluster_dist = df['Cluster'].value_counts().sort_index()
                    st.bar_chart(cluster_dist)
                
                with col4:
                    st.markdown("### Estatísticas por Cluster")
                    stats = df.groupby('Cluster')[selected_features].mean()
                    st.dataframe(stats.style.format("{:.2f}").background_gradient(cmap='Blues'))
                
                # Visualização interativa
                st.subheader("Visualização Interativa")
                x_axis = st.selectbox("Eixo X:", selected_features, index=0)
                y_axis = st.selectbox("Eixo Y:", selected_features, index=1)
                
                palette = sns.color_palette("husl", n_clusters)
                visualizer.plot_scatter(df, x_axis, y_axis, 'Cluster', palette)
                
                # Análise de silhueta detalhada
                st.subheader("Análise de Silhueta por Cluster")
                analyzer.plot_silhouette_analysis(X, labels)
                
            except Exception as e:
                st.error(f"Erro na análise: {e}")
        else:
            st.warning("Selecione pelo menos 2 variáveis para realizar o clustering.")

if __name__ == "__main__":
    main()

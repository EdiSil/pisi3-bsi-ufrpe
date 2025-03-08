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

# Configuração de estilo profissional atualizada
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
        self.LABEL_MAP = {
            'quilometragem': 'QUILOMETRAGEM (Km)',
            'preco': 'PREÇO (R$)',
            'ano': 'ANO DE FABRICAÇÃO',
            'full_range': 'AUTONOMIA (Km)',
            'Car Age': 'IDADE DO VEÍCULO (Anos)'
        }

    def prepare_data(self, features):
        try:
            self.features = features
            X = self.data[features]
            return self.scaler.fit_transform(X)
        except KeyError as e:
            st.error(f"VARIÁVEL NÃO ENCONTRADA NO DATASET: {e}")
            return None

    def calculate_elbow(self, X, max_clusters=10):
        inertia = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
        return inertia

    def calculate_silhouette(self, X, max_clusters=10):
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
        try:
            self.n_clusters = n_clusters
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            return self.kmeans.fit_predict(X)
        except ValueError as e:
            st.error(f"ERRO NO CLUSTERING: {e}")
            return None

    def plot_silhouette_analysis(self, X, labels):
        try:
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

            ax.set_title("ANÁLISE DE SILHUETA POR CLUSTER", fontweight='bold', pad=15)
            ax.set_xlabel("COEFICIENTE DE SILHUETA", fontweight='bold')
            ax.set_ylabel("CLUSTER", fontweight='bold')
            ax.axvline(x=silhouette_avg, color="red", linestyle="--")
            ax.set_yticks([])
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"ERRO NA ANÁLISE DE SILHUETA: {e}")

class ClusterVisualizer:
    def __init__(self):
        self.LABEL_MAP = {
            'quilometragem': 'QUILOMETRAGEM (Km)',
            'preco': 'PREÇO (R$)',
            'ano': 'ANO DE FABRICAÇÃO',
            'full_range': 'AUTONOMIA (Km)',
            'Car Age': 'IDADE DO VEÍCULO (Anos)'
        }

    def format_thousands(self, x, pos):
        return f'{x:,.0f}'.replace(",", ".")

    def format_reais(self, x, pos):
        return f'R$ {x:,.0f}'.replace(",", ".")

    def plot_elbow(self, inertia, max_clusters):
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(x=range(1, max_clusters + 1), y=inertia, marker='o', ax=ax)
            
            ax.set_title('MÉTODO DO COTOVELO - SELEÇÃO DE CLUSTERS',
                        fontsize=12, fontweight='bold', pad=15)
            ax.set_xlabel('NÚMERO DE CLUSTERS', fontweight='bold')
            ax.set_ylabel('INÉRCIA', fontweight='bold')
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(self.format_thousands))
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"ERRO AO PLOTAR MÉTODO DO COTOVELO: {e}")

    def plot_silhouette_scores(self, scores, max_clusters):
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(x=range(2, max_clusters + 1), y=scores, marker='o', ax=ax)
            
            ax.set_title('PONTUAÇÃO MÉDIA DE SILHUETA',
                       fontsize=12, fontweight='bold', pad=15)
            ax.set_xlabel('NÚMERO DE CLUSTERS', fontweight='bold')
            ax.set_ylabel('SCORE DE SILHUETA', fontweight='bold')
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"ERRO AO PLOTAR SCORES DE SILHUETA: {e}")

    def plot_scatter(self, data, x_col, y_col, hue_col, palette):
        """Gráfico de dispersão K-Means com formatação em Reais e escalas específicas"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plotagem principal sem legenda
            sns.scatterplot(
                data=data, 
                x=x_col, 
                y=y_col, 
                hue=hue_col,
                palette=palette, 
                s=60, 
                ax=ax, 
                edgecolor='w', 
                linewidth=0.5,
                legend=False
            )

            # Configuração personalizada para eixos de preço
            def configure_price_axis(axis, data_col):
                axis.set_major_formatter(mticker.FuncFormatter(self.format_reais))
                
                # Escalas específicas em Reais
                ticks = [10000, 50000, 100000, 200000, 300000, 400000, 500000]
                valid_ticks = [t for t in ticks if t <= data[data_col].max()]
                
                axis.set_ticks(valid_ticks)
                axis.set_ticklabels([self.format_reais(t, None) for t in valid_ticks])

            # Configurar eixos
            if x_col == 'preco':
                configure_price_axis(ax.xaxis, x_col)
            else:
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(self.format_thousands))

            if y_col == 'preco':
                configure_price_axis(ax.yaxis, y_col)
            else:
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(self.format_thousands))

            ax.set_title(f'K-MEANS CLUSTERING: {self.LABEL_MAP[x_col]} vs {self.LABEL_MAP[y_col]}',
                        fontsize=12, fontweight='bold', pad=15)
            ax.set_xlabel(self.LABEL_MAP[x_col], fontweight='bold')
            ax.set_ylabel(self.LABEL_MAP[y_col], fontweight='bold')
            
            ax.tick_params(axis='both', labelsize=8)
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"ERRO AO GERAR GRÁFICO DE DISPERSÃO: {e}")

def main():
    st.set_page_config(page_title="Análise de Clusters de Carros", layout="wide")
    st.title("ANÁLISE INTERATIVA DOS CLUSTERS ")
    
    file_path = 'Datas/2_Cars_clusterizado.csv'
    
    if not os.path.exists(file_path):
        st.error(f"ARQUIVO NÃO ENCONTRADO: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
        st.session_state['data'] = df
        st.success("DADOS CARREGADOS COM SUCESSO!")
    except Exception as e:
        st.error(f"ERRO AO CARREGAR DADOS: {e}")
        return

    st.sidebar.header("CONFIGURAÇÕES")
    available_features = ['quilometragem', 'preco', 'ano', 'full_range', 'Car Age']
    FEATURE_LABELS = {
        'quilometragem': 'QUILOMETRAGEM',
        'preco': 'PREÇO',
        'ano': 'ANO',
        'full_range': 'AUTONOMIA',
        'Car Age': 'IDADE DO VEÍCULO'
    }
    
    selected_features = st.sidebar.multiselect(
        "SELECIONE AS VARIÁVEIS PARA CLUSTERING:",
        options=available_features,
        default=['quilometragem', 'preco', 'ano'],
        format_func=lambda x: FEATURE_LABELS[x]
    )
    
    max_clusters_elbow = st.sidebar.slider(
        "SELECIONE NÚMERO MÁXIMO DE CLUSTERS (COTOVELO):",
        2, 15, 15
    )
    
    max_clusters_silhouette = st.sidebar.slider(
        "SELECIONE NÚMERO MÁXIMO DE CLUSTERS (SILHUETA):",
        2, 15, 15
    )
    
    n_clusters = st.sidebar.slider(
        "SELECIONE NÚMERO DE CLUSTERS (VISUALIZAÇÃO INTERATIVA):",
        2, 15, 15
    )
    
    analyzer = CarClusterAnalysis(df)
    visualizer = ClusterVisualizer()
    
    with st.container():
        if len(selected_features) >= 2:
            try:
                X = analyzer.prepare_data(selected_features)
                if X is None:
                    return
                
                st.subheader("MÉTODO DO COTOVELO")
                inertia = analyzer.calculate_elbow(X, max_clusters_elbow)
                visualizer.plot_elbow(inertia, max_clusters_elbow)
                
                st.subheader("ANÁLISE DE SILHUETA")
                silhouette_scores = analyzer.calculate_silhouette(X, max_clusters_silhouette)
                visualizer.plot_silhouette_scores(silhouette_scores, max_clusters_silhouette)
                
                labels = analyzer.perform_clustering(X, n_clusters)
                if labels is None:
                    return
                df['Cluster'] = labels

                st.subheader("VISUALIZAÇÃO INTERATIVA")
                col1, col2 = st.columns(2)
                
                with col1:
                    x_axis = st.selectbox("EIXO X:", selected_features, index=0,
                                        format_func=lambda x: FEATURE_LABELS[x])
                with col2:
                    y_axis = st.selectbox("EIXO Y:", selected_features, index=1,
                                        format_func=lambda x: FEATURE_LABELS[x])
                
                palette = sns.color_palette("husl", n_clusters)
                visualizer.plot_scatter(df, x_axis, y_axis, 'Cluster', palette)
                
                st.subheader("ANÁLISE DE SILHUETA POR CLUSTER")
                analyzer.plot_silhouette_analysis(X, labels)
                
            except Exception as e:
                st.error(f"ERRO NA ANÁLISE: {e}")
        else:
            st.warning("SELECIONE PELO MENOS 2 VARIÁVEIS PARA REALIZAR O CLUSTERING!")

if __name__ == "__main__":
    main()

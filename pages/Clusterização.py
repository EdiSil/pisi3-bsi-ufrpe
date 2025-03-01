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

    # ... (mantidos todos os métodos anteriores da classe CarClusterAnalysis)

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

    def format_year(self, x, pos):
        return f'{int(x)}'  # Formatação correta para anos

    # ... (mantidos todos os métodos anteriores da classe ClusterVisualizer até plot_cluster_distribution)

    def plot_scatter(self, data, x_col, y_col, hue_col, palette):
        """Plota gráfico de dispersão com formatação profissional"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Converter anos para inteiros para evitar decimais
            if x_col == 'ano':
                data[x_col] = data[x_col].astype(int)
            if y_col == 'ano':
                data[y_col] = data[y_col].astype(int)

            sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col,
                           palette=palette, s=60, ax=ax, edgecolor='w', linewidth=0.5)

            # Configuração do eixo X
            if x_col == 'preco':
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(self.format_reais))
            elif x_col == 'ano':
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(self.format_year))
            else:
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(self.format_thousands))

            # Configuração do eixo Y
            if y_col == 'preco':
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(self.format_reais))
            elif y_col == 'ano':
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(self.format_year))
            else:
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(self.format_thousands))

            ax.set_title(f'CLUSTERS: {self.LABEL_MAP[x_col]} vs {self.LABEL_MAP[y_col]}',
                        fontsize=12, fontweight='bold', pad=15)
            ax.set_xlabel(self.LABEL_MAP[x_col], fontweight='bold')
            ax.set_ylabel(self.LABEL_MAP[y_col], fontweight='bold')
            
            ax.legend(title='CLUSTERS', 
                     bbox_to_anchor=(1.05, 1), 
                     loc='upper left',
                     title_fontproperties={'weight':'bold'})
            
            ax.tick_params(axis='both', labelsize=8)
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"ERRO AO GERAR GRÁFICO DE DISPERSÃO: {e}")

def main():
    # ... (mantido o mesmo código anterior da função main até a seção de controles da sidebar)
    
    # Sidebar controls
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
    
    # ... (mantido o restante do código da função main)

if __name__ == "__main__":
    main()

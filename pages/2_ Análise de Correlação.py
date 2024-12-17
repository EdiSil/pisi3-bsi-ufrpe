import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import numpy as np

class CarAnalysisApp:
    def __init__(self, file_url):
        self.file_url = file_url
        self.df = None
        self.load_data()

    def load_data(self):
        """Carregar os dados CSV diretamente da URL"""
        self.df = pd.read_csv(self.file_url)

        # Convertemos para valores numéricos para análise de correlação
        self.df['preco'] = pd.to_numeric(self.df['preco'], errors='coerce')
        self.df['ano'] = pd.to_numeric(self.df['ano'], errors='coerce')
        self.df['quilometragem'] = pd.to_numeric(self.df['quilometragem'], errors='coerce')

    def correlation_matrix(self):
        """Calcula a matriz de correlação"""
        correlation_cols = ['preco', 'ano', 'quilometragem']
        correlation_matrix = self.df[correlation_cols].corr()

        # Gerar Heatmap interativo usando Plotly
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='Viridis',
            colorbar=dict(title="Correlação")
        ))

        fig.update_layout(
            title="Matriz de Correlação",
            xaxis_title="Variáveis",
            yaxis_title="Variáveis",
            template="plotly_dark"
        )
        return fig

    def show_app(self):
        """Exibe a interface do Streamlit"""
        st.title("Análise de Dados de Veículos")
        st.markdown("""
            Neste aplicativo, você pode visualizar a matriz de correlação entre as variáveis `preco`, `ano` e `quilometragem`.
        """)

        st.subheader('Matriz de Correlação e Heatmap')

        # Exibir a Matriz de Correlação com Heatmap
        correlation_fig = self.correlation_matrix()
        st.plotly_chart(correlation_fig, use_container_width=True)

# URL do arquivo CSV
file_url = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_dataset002.csv"

# Criar e executar a aplicação
app = CarAnalysisApp(file_url)
app.show_app()

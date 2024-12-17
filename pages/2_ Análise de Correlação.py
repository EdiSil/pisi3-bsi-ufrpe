import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

class CarAnalysisApp:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.load_data()
        self.clean_data()

    def load_data(self):
        """Carrega os dados CSV a partir de uma URL ou caminho local."""
        return pd.read_csv(self.file_path)

    def clean_data(self):
        """Limpa os dados, convertendo as colunas para os tipos apropriados."""
        # Conversão das colunas numéricas
        self.df['preco'] = pd.to_numeric(self.df['preco'], errors='coerce')
        self.df['ano'] = pd.to_numeric(self.df['ano'], errors='coerce')
        self.df['quilometragem'] = pd.to_numeric(self.df['quilometragem'], errors='coerce')

        # Remover linhas com valores nulos em colunas importantes
        self.df.dropna(subset=['marca', 'modelo', 'preco', 'ano', 'quilometragem', 'combustivel', 'tipo'], inplace=True)

    def plot_correlation_matrix(self):
        """Exibe a matriz de correlação interativa utilizando Plotly."""
        correlation_matrix = self.df[['marca', 'modelo', 'ano', 'quilometragem', 'preco', 'combustivel', 'tipo']].corr()
        
        # Gerar heatmap interativo
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='Viridis',
            zmin=-1, zmax=1
        ))
        
        fig.update_layout(
            title='Matriz de Correlação Interativa',
            xaxis_title='Variáveis',
            yaxis_title='Variáveis',
            template='plotly_dark'
        )

        return fig

    def plot_price_distribution_by_fuel_type(self):
        """Exibe o gráfico de distribuição de preço por tipo de combustível."""
        fig = px.box(self.df, x='combustivel', y='preco', title='Distribuição de Preço por Tipo de Combustível')
        fig.update_layout(template='plotly_dark')
        return fig

    def run_app(self):
        """Executa o aplicativo Streamlit."""
        st.title("Análise de Carros")

        # Exibe os dados carregados
        st.write("Dados Carregados:")
        st.dataframe(self.df.head())

        # Matriz de Correlação
        st.header("Matriz de Correlação")
        correlation_fig = self.plot_correlation_matrix()
        st.plotly_chart(correlation_fig)

        # Gráfico de Distribuição de Preço por Combustível
        st.header("Distribuição de Preço por Tipo de Combustível")
        fuel_price_fig = self.plot_price_distribution_by_fuel_type()
        st.plotly_chart(fuel_price_fig)


# URL do arquivo CSV
file_path = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_dataset002.csv"

# Inicializa o aplicativo e executa
app = CarAnalysisApp(file_path)
app.run_app()

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

class CorrelationAnalysisApp:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.clean_data()

    def clean_data(self):
        # Codificando variáveis categóricas
        label_encoder = LabelEncoder()
        categorical_columns = ['marca', 'modelo', 'combustivel', 'tipo']
        for column in categorical_columns:
            if column in self.df.columns:
                self.df[column] = label_encoder.fit_transform(self.df[column].astype(str))
        # Filtrando para manter apenas as colunas de interesse
        self.df = self.df[['marca', 'modelo', 'ano', 'quilometragem', 'preco', 'combustivel', 'tipo']]

    def plot_correlation_matrix(self):
        # Calcular a matriz de correlação entre as variáveis numéricas
        correlation_matrix = self.df.corr()
        
        # Criar um heatmap interativo usando Plotly
        fig = px.imshow(correlation_matrix,
                        labels=dict(x="Variáveis", y="Variáveis", color="Correlação"),
                        color_continuous_scale='pinkyl', # Usando o gradiente de pink a roxo
                        zmin=-1, zmax=1, # Ajustando a escala de correlação
                        title="Matriz de Correlação")
        return fig

    def run_app(self):
        st.title('Análise de Correlação - OLX Cars Dataset')

        # Exibir os dados
        st.header("Dados Carregados:")
        st.dataframe(self.df.head())

        # Plotar matriz de correlação
        st.header("Matriz de Correlação Interativa")
        correlation_fig = self.plot_correlation_matrix()
        st.plotly_chart(correlation_fig)

# URL do arquivo CSV
file_url = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_dataset002.csv"
app = CorrelationAnalysisApp(file_url)
app.run_app()

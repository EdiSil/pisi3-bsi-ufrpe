import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

class CarDataAnalysis:
    def __init__(self, data_url):
        self.data_url = data_url
        self.df = self.load_data()

    # Função para carregar os dados
    def load_data(self):
        try:
            # Carregar dados diretamente da URL
            df = pd.read_csv(self.data_url)
            # Limpar e tratar dados
            df = self.clean_data(df)
            return df
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {e}")
            return None

    # Função para limpar e preparar os dados
    def clean_data(self, df):
        # Selecionar apenas as colunas necessárias
        df = df[['marca', 'modelo', 'ano', 'quilometragem', 'preco', 'combustivel', 'tipo']]

        # Converter colunas para numéricas onde for necessário
        df['ano'] = pd.to_numeric(df['ano'], errors='coerce')
        df['preco'] = pd.to_numeric(df['preco'], errors='coerce')
        df['quilometragem'] = pd.to_numeric(df['quilometragem'], errors='coerce')

        # Remover linhas com valores nulos
        df.dropna(inplace=True)

        return df

    # Função para calcular a matriz de correlação
    def calculate_correlation(self):
        # Selecionar apenas colunas numéricas para a correlação
        df_corr = self.df[['ano', 'quilometragem', 'preco']]
        correlation_matrix = df_corr.corr()
        return correlation_matrix

    # Função para exibir o heatmap interativo
    def plot_correlation_matrix(self):
        # Calcular a matriz de correlação
        correlation_matrix = self.calculate_correlation()

        # Plotly Heatmap
        fig = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns.values,
            y=correlation_matrix.columns.values,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            showscale=True
        )
        fig.update_layout(title='Matriz de Correlação', width=800, height=600)
        st.plotly_chart(fig)

# Configuração da aplicação Streamlit
st.title('Análise de Correlação - OLX Carros')

# Carregar dados a partir da URL
data_url = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_dataset002.csv"
car_analysis = CarDataAnalysis(data_url)

# Exibir os dados carregados
st.subheader('Pré-visualização dos Dados:')
st.dataframe(car_analysis.df.head())

# Gerar a Matriz de Correlação e Heatmap
st.subheader('Matriz de Correlação e Heatmap')
car_analysis.plot_correlation_matrix()

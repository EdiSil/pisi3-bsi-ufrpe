import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import streamlit as st
import numpy as np

# Classe para Análise de Carros
class CarAnalysis:
    def __init__(self, data_url):
        # Carregar dados diretamente da URL
        self.df = pd.read_csv(data_url)
        # Limpeza dos dados
        self.df = self.clean_data()

    def clean_data(self):
        # Limpeza de dados: remover valores nulos e garantir que as colunas relevantes sejam numéricas
        relevant_columns = ['marca', 'modelo', 'ano', 'quilometragem', 'preco', 'combustivel', 'tipo']
        # Remover colunas não relevantes
        self.df = self.df[relevant_columns]
        
        # Converter colunas numéricas para o tipo correto
        self.df['ano'] = pd.to_numeric(self.df['ano'], errors='coerce')
        self.df['quilometragem'] = pd.to_numeric(self.df['quilometragem'], errors='coerce')
        self.df['preco'] = pd.to_numeric(self.df['preco'], errors='coerce')

        # Remover linhas com valores nulos
        self.df = self.df.dropna(subset=['ano', 'quilometragem', 'preco'])

        return self.df

    # Método para plotar a matriz de correlação interativa
    def plot_correlation_matrix(self):
        # Selecionar as colunas numéricas para a análise de correlação
        corr_columns = ['ano', 'quilometragem', 'preco']
        correlation_matrix = self.df[corr_columns].corr()

        # Criando o heatmap interativo com Plotly
        fig = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=corr_columns,
            y=corr_columns,
            colorscale='RdBu',
            showscale=True
        )

        # Atualizar escala de correlação (valores personalizados)
        fig.update_traces(colorscale='RdBu', zmin=-1.0, zmax=1.0, colorbar=dict(
            tickvals=[-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0],
            ticktext=['-1.0', '-0.75', '-0.5', '-0.25', '0', '0.25', '0.5', '0.75', '1.0']
        ))

        fig.update_layout(
            title="Matriz de Correlação",
            xaxis_title="Variáveis",
            yaxis_title="Variáveis",
            template="plotly_white"
        )

        st.plotly_chart(fig)

    # Método para exibir o gráfico de dispersão interativo (Preço x Ano por Marca)
    def plot_interactive_scatter(self):
        fig = px.scatter(self.df, x='ano', y='preco', color='marca', 
                         hover_data=['modelo', 'combustivel', 'tipo'], 
                         title="Preço x Ano por Marca")
        st.plotly_chart(fig)

    # Método para exibir o histograma interativo entre Preço e Combustível por Marca
    def plot_interactive_histogram_fuel(self):
        # Alterar a paleta de cores para Gasolina (Roxa) e GNV (Pink)
        fig = px.histogram(self.df, x='preco', color='combustivel', barmode='overlay', 
                           title="Histograma: Preço x Combustível por Marca",
                           color_discrete_map={"Gasolina": "purple", "GNV": "pink", "Álcool": "blue", "Diesel": "green"})
        fig.update_layout(
            xaxis_title="Preço",
            yaxis_title="Contagem",
            template="plotly_white"
        )
        st.plotly_chart(fig)

# Configuração da aplicação Streamlit
def run_app():
    # URL do arquivo CSV
    DATA_URL = "https://github.com/EdiSil/pisi3-bsi-ufrpe/raw/main/data/OLX_cars_dataset002.csv"
    
    # Criação do objeto de análise
    car_analysis = CarAnalysis(DATA_URL)

    # Exibição do título da aplicação
    st.title("Análise de Correlação e Preços de Carros")

    # Plotar a matriz de correlação
    st.header("Matriz de Correlação")
    car_analysis.plot_correlation_matrix()

    # Plotar o gráfico de dispersão interativo (Preço x Ano por Marca)
    st.header("Gráfico Interativo: Preço x Ano por Marca")
    car_analysis.plot_interactive_scatter()

    # Plotar o histograma interativo (Preço x Combustível por Marca)
    st.header("Histograma Interativo: Preço x Combustível por Marca")
    car_analysis.plot_interactive_histogram_fuel()

if __name__ == "__main__":
    run_app()

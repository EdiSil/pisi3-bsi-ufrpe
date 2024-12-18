import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import streamlit as st
import numpy as np

# Classe para Análise de Carros
class CarAnalysis:
    def __init__(self, data_url):
        """
        Inicializa a classe com o URL dos dados CSV.
        """
        self.df = pd.read_csv(data_url)
        self.df = self.clean_data()

    def clean_data(self):
        """
        Realiza a limpeza dos dados: remove valores nulos e converte colunas numéricas.
        """
        relevant_columns = ['marca', 'modelo', 'ano', 'quilometragem', 'preco', 'combustivel', 'tipo']
        self.df = self.df[relevant_columns]

        # Converte para numérico e trata erros com 'coerce' (transforma erros em NaN)
        self.df['ano'] = pd.to_numeric(self.df['ano'], errors='coerce')
        self.df['quilometragem'] = pd.to_numeric(self.df['quilometragem'], errors='coerce')
        self.df['preco'] = pd.to_numeric(self.df['preco'], errors='coerce')

        # Remove as linhas com valores nulos nas colunas numéricas
        self.df = self.df.dropna(subset=['ano', 'quilometragem', 'preco'])

        return self.df

    def plot_correlation_matrix(self):
        """
        Plota a matriz de correlação entre as colunas numéricas: 'ano', 'quilometragem', e 'preco'.
        """
        corr_columns = ['ano', 'quilometragem', 'preco']
        correlation_matrix = self.df[corr_columns].corr()

        # Criar o heatmap interativo
        fig = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=corr_columns,
            y=corr_columns,
            colorscale='RdBu',
            showscale=True
        )

        # Atualizar as propriedades do gráfico
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

    def plot_interactive_scatter(self):
        """
        Exibe um gráfico de dispersão interativo entre 'ano' e 'preco' categorizado por 'marca'.
        """
        fig = px.scatter(self.df, x='ano', y='preco', color='marca',
                         hover_data=['modelo', 'combustivel', 'tipo'],
                         title="Preço x Ano por Marca")
        st.plotly_chart(fig)

    def plot_interactive_histogram(self):
        """
        Exibe um histograma interativo mostrando a relação entre 'preco' e 'combustivel' por 'marca'.
        """
        df_filtered = self.df[['marca', 'preco', 'combustivel']]

        # Paleta de cores personalizada (do rosa ao roxo)
        color_map = {
            'Gasolina': '#FF66B2',  # Cor rosa
            'GNV': '#800080'  # Cor roxa
        }

        # Criando o histograma
        fig = px.histogram(df_filtered, x="combustivel", y="preco", color="marca",
                           category_orders={"combustivel": ['Gasolina', 'GNV']},
                           title="Histograma: Preço x Combustível por Marca",
                           color_discrete_map=color_map, barmode='group')

        # Customizações para o layout
        fig.update_layout(
            title="Histograma: Preço x Combustível por Marca",
            xaxis_title="Combustível",
            yaxis_title="Preço",
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12, color="black"),
            title_x=0.5,  # Centralizando o título
            showlegend=True,
            barmode='group',  # Barras lado a lado
            margin=dict(l=40, r=40, t=40, b=40)  # Margens para maior clareza
        )

        # Personalização do hover
        fig.update_traces(
            hovertemplate="<b>Marca:</b> %{marker.color}<br><b>Preço:</b> %{y}<br><b>Combustível:</b> %{customdata[0]}<br><b>Modelo:</b> %{customdata[1]}<br><b>Tipo:</b> %{customdata[2]}"
        )

        st.plotly_chart(fig)

# Função principal para rodar a aplicação Streamlit
def run_app():
    # URL do arquivo CSV no GitHub
    DATA_URL = "https://github.com/EdiSil/pisi3-bsi-ufrpe/raw/main/data/OLX_cars_dataset002.csv"
    
    # Criação do objeto de análise de carros
    car_analysis = CarAnalysis(DATA_URL)

    # Exibição do título da aplicação
    st.title("Análise de Correlação e Preços de Carros")

    # Plotar a matriz de correlação
    st.header("Matriz de Correlação")
    car_analysis.plot_correlation_matrix()

    # Plotar o gráfico de dispersão interativo
    st.header("Gráfico Interativo: Preço x Ano por Marca")
    car_analysis.plot_interactive_scatter()

    # Plotar o histograma interativo
    st.header("Histograma: Preço x Combustível por Marca")
    car_analysis.plot_interactive_histogram()

if __name__ == "__main__":
    run_app()

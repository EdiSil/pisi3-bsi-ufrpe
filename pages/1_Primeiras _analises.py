import streamlit as st
import pandas as pd
import plotly.express as px
from utils.build import build_header, breakrows, top_categories
from utils.charts import boxplot, scatter, treemap, hist, bar, select_chart


class DataAnalysisApp:
    """
    Classe principal para executar a aplicação de análise exploratória de dados.
    """

    def __init__(self, data_url):
        """
        Inicializa a classe com o caminho do dataset.
        """
        self.data_url = data_url
        self.data = None
        self.data_group = None
        self.data_filtered = None

    def load_data(self):
        """
        Carrega os dados do arquivo CSV a partir do GitHub.
        """
        @st.cache_data
        def load_csv(url):
            return pd.read_csv(url)

        self.data = load_csv(self.data_url)
        st.success("Dados carregados com sucesso!")

    def prepare_data(self):
        """
        Prepara os dados realizando agrupamentos e filtragens necessárias.
        """
        # Agrupamento
        self.data_group = (
            self.data.groupby(['preco', 'marca', 'ano', 'modelo', 'combustivel', 'tipo', 'quilometragem'])
            .size()
            .reset_index(name='Total')
        )
        self.data_group.sort_values('Total', ascending=True, inplace=True)

        # Filtragem das top 10 marcas
        self.data_filtered = top_categories(data=self.data, top=10, label='marca')

    def display_header(self):
        """
        Exibe o cabeçalho da página.
        """
        build_header(
            title='Primeiras Análises',
            hdr='# PRIMEIRAS ANÁLISES E VISUALIZAÇÕES',
            p='''<p>Vamos realizar as primeiras observações dos dados e suas correlações entre algumas variáveis</p>'''
        )

    def display_boxplot(self):
        """
        Exibe um boxplot das marcas por quilometragem.
        """
        breakrows()
        boxplot(
            data=self.data_filtered,
            title='BoxPlot da Marca por Quilometragem',
            x='marca',
            y='quilometragem',
            p="""
                <p style='text-align:justify;'> As marcas têm uma concentração entre 20k e 60k quilômetros rodados.
                Algumas marcas como Hyundai, Nissan, Jeep e BMW têm veículos passando dos 100k de quilometragem.</p>
            """
        )

    def display_histogram(self):
        """
        Exibe um histograma da quantidade de veículos por marca.
        """
        breakrows()
        hist(
            title='Histograma da Marca',
            data=self.data,
            x='marca'
        )

    def display_bar_chart(self):
        """
        Exibe um gráfico de barras da relação entre preço e ano.
        """
        breakrows()
        data_ano = self.data.groupby(['ano'])['preco'].size().reset_index()
        bar(
            title='Gráfico de Barras: Preço x Ano',
            data=data_ano,
            x='ano',
            y='preco',
            p="""
                <p>Observamos que os preços dos veículos tendem a ser mais caros
                com a variação de ano entre 2014 e 2016.</p>
            """
        )

    def display_scatter_charts(self):
        """
        Exibe gráficos de dispersão interativos.
        """
        breakrows()
        select_chart(
            self.data,
            x='marca',
            options=self.data.columns,
            type_graph=px.scatter,
            type_txt='Distribuição da'
        )
        breakrows()
        select_chart(
            self.data_group,
            x='quilometragem',
            options=self.data.columns,
            type_graph=px.scatter,
            type_txt='Distribuição da'
        )

    def run(self):
        """
        Executa todos os métodos para rodar a aplicação.
        """
        self.display_header()
        self.load_data()
        self.prepare_data()
        self.display_boxplot()
        self.display_histogram()
        self.display_bar_chart()
        self.display_scatter_charts()


# URL do dataset
DATA_URL = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_dataset002.csv"

# Inicializa e executa o aplicativo
if __name__ == "__main__":
    app = DataAnalysisApp(DATA_URL)
    app.run()

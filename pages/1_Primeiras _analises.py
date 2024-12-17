import streamlit as st
import pandas as pd
import plotly.express as px


class DataAnalysisApp:
    """
    Classe principal para executar a aplicação de análise exploratória de dados.
    """
    def __init__(self, data_path):
        """
        Inicializa a classe com o caminho do dataset.
        """
        self.data_path = data_path
        self.data = None
        self.data_group = None
        self.data_filtered = None

    def load_data(self):
        """
        Carrega os dados do arquivo CSV.
        """
        try:
            self.data = pd.read_csv(self.data_path)
            st.success("✅ Dados carregados com sucesso!")
        except Exception as e:
            st.error(f"❌ Erro ao carregar o arquivo: {e}")
            st.stop()

    def prepare_data(self):
        """
        Realiza os agrupamentos e filtragens necessários.
        """
        # Agrupamento com base em colunas definidas
        self.data_group = (
            self.data.groupby(['preco', 'marca', 'ano', 'modelo', 'combustivel', 'tipo', 'quilometragem'])
            .size()
            .reset_index(name='Total')
        )
        self.data_group.sort_values('Total', ascending=True, inplace=True)

        # Filtragem das top 10 marcas
        self.data_filtered = self.top_categories(data=self.data, top=10, label='marca')

    @staticmethod
    def top_categories(data, top, label):
        """
        Filtra as 'top N' categorias de uma determinada coluna.
        """
        top_data = data[label].value_counts().head(top).index
        return data[data[label].isin(top_data)]

    def display_header(self):
        """
        Exibe o cabeçalho da aplicação.
        """
        st.title("Primeiras Análises")
        st.header("# PRIMEIRAS ANÁLISES E VISUALIZAÇÕES")
        st.markdown(
            """
            <p>Vamos realizar as primeiras observações dos dados e suas correlações 
            entre algumas variáveis para extrair informações úteis.</p>
            """,
            unsafe_allow_html=True,
        )

    def display_boxplot(self):
        """
        Exibe um Boxplot das marcas por quilometragem.
        """
        st.subheader("BoxPlot da Marca por Quilometragem")
        fig = px.box(
            self.data_filtered,
            x='marca',
            y='quilometragem',
            title='BoxPlot: Quilometragem por Marca',
            color='marca'
        )
        st.plotly_chart(fig)
        st.markdown(""" 
        <p style='text-align:justify;'>As marcas têm uma concentração entre 20k e 60k quilômetros rodados. 
        Algumas marcas como Hyundai, Nissan, Jeep e BMW têm veículos acima de 100k de quilometragem.</p>
        """, unsafe_allow_html=True)

    def display_histogram(self):
        """
        Exibe um histograma da quantidade de veículos por marca.
        """
        st.subheader("Histograma da Quantidade de Veículos por Marca")
        fig = px.histogram(
            self.data,
            x='marca',
            title='Histograma: Quantidade de Veículos por Marca',
            color='marca'
        )
        st.plotly_chart(fig)

    def display_bar_chart(self):
        """
        Exibe um gráfico de barras da relação entre preço e ano.
        """
        st.subheader("Gráfico de Barras: Preço x Ano")
        data_ano = self.data.groupby(['ano'])['preco'].mean().reset_index()
        fig = px.bar(
            data_ano,
            x='ano',
            y='preco',
            title='Gráfico de Barras: Média de Preço por Ano'
        )
        st.plotly_chart(fig)
        st.markdown(""" 
        <p style='text-align:justify;'>Observamos que os preços médios dos veículos tendem a ser mais elevados 
        nos anos recentes entre 2014 e 2016.</p>
        """, unsafe_allow_html=True)

    def display_scatter_charts(self):
        """
        Exibe gráficos de dispersão interativos.
        """
        st.subheader("Gráfico de Dispersão Interativo")
        fig = px.scatter(
            self.data,
            x='marca',
            y='quilometragem',
            color='preco',
            title="Dispersão: Quilometragem por Marca (Colorido por Preço)"
        )
        st.plotly_chart(fig)

        st.subheader("Gráfico de Dispersão da Quilometragem")
        fig_group = px.scatter(
            self.data_group,
            x='quilometragem',
            y='preco',
            size='Total',
            color='marca',
            title="Dispersão: Quilometragem x Preço (Agrupado)"
        )
        st.plotly_chart(fig_group)

    def run(self):
        """
        Executa todos os métodos necessários para rodar a aplicação.
        """
        self.display_header()
        self.load_data()
        self.prepare_data()
        self.display_boxplot()
        self.display_histogram()
        self.display_bar_chart()
        self.display_scatter_charts()


# Caminho do dataset local
DATA_PATH = "/mnt/data/OLX_cars_dataset002.csv"

# Inicializa e executa o aplicativo
if __name__ == "__main__":
    app = DataAnalysisApp(DATA_PATH)
    app.run()

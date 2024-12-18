import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import streamlit as st
import numpy as np

# Classe para Análise de Carros
class CarAnalysis:
    def __init__(self, data_url, exchange_rate):
        """
        Inicializa a classe com o URL dos dados CSV e a taxa de câmbio.
        """
        self.data_url = data_url
        self.exchange_rate = exchange_rate
        self.df = pd.read_csv(data_url)
        self.df = self.clean_data()

    def clean_data(self):
        """
        Realiza a limpeza dos dados: remove valores nulos, converte colunas numéricas
        e aplica conversão de moeda no preço.
        """
        relevant_columns = ['marca', 'modelo', 'ano', 'quilometragem', 'preco', 'combustivel', 'tipo']
        self.df = self.df[relevant_columns]

        # Converte para numérico e trata erros com 'coerce' (transforma erros em NaN)
        self.df['ano'] = pd.to_numeric(self.df['ano'], errors='coerce')
        self.df['quilometragem'] = pd.to_numeric(self.df['quilometragem'], errors='coerce')
        self.df['preco'] = pd.to_numeric(self.df['preco'], errors='coerce')

        # Remove as linhas com valores nulos nas colunas numéricas
        self.df = self.df.dropna(subset=['ano', 'quilometragem', 'preco'])

        # Converte preço de Rúpia Paquistanesa para Real Brasileiro
        self.df['preco'] = self.df['preco'] * self.exchange_rate

        return self.df

    def plot_correlation_matrix(self):
        """
        Plota a matriz de correlação entre as colunas numéricas: 'ano', 'quilometragem', e 'preco'.
        """
        corr_columns = ['ano', 'quilometragem', 'preco']
        correlation_matrix = self.df[corr_columns].corr()

        # Criar o heatmap interativo
        fig = ff.create_annotated_heatmap(
            z=np.round(correlation_matrix.values, 2),  # Duas casas decimais no eixo Z
            x=corr_columns,
            y=corr_columns,
            colorscale='RdBu',
            showscale=True
        )

        # Atualizar as propriedades do gráfico
        fig.update_traces(colorscale='RdBu', zmin=-1.0, zmax=1.0)

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
        fig = px.scatter(
            self.df, x='ano', y='preco', color='marca',
            hover_data=['modelo', 'combustivel', 'tipo'],
            title="Preço x Ano por Marca"
        )
        st.plotly_chart(fig)

    def plot_interactive_histogram(self):
        """
        Exibe um histograma interativo mostrando a relação entre 'preco' e 'combustivel' por 'marca'.
        """
        # Agrupando dados para garantir que as somas sejam representadas corretamente
        df_grouped = self.df.groupby(['combustivel', 'marca'], as_index=False)['preco'].sum()

        # Criando o histograma
        fig = px.histogram(
            df_grouped, x="combustivel", y="preco", color="marca",
            title="Histograma: Preço x Combustível por Marca",
            barmode='group'
        )

        # Customizações para o layout
        fig.update_layout(
            title="Histograma: Preço x Combustível por Marca",
            xaxis_title="Combustível",
            yaxis_title="Soma do preço",
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12, color="black"),
            title_x=0.5,
            showlegend=True,
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # Adicionando hover com detalhes corretos
        fig.update_traces(
            hovertemplate="<b>Marca:</b> %{customdata[1]}<br>"
                          "<b>Soma do preço:</b> R$ %{y:.2f}<br>"
                          "<b>Combustível:</b> %{x}",
            customdata=df_grouped[['combustivel', 'marca']].values  # Passando comb. e marca ao hover
        )

        st.plotly_chart(fig)

# Função principal para rodar a aplicação Streamlit
def run_app():
    # URL do arquivo CSV no GitHub
    DATA_URL = "https://github.com/EdiSil/pisi3-bsi-ufrpe/raw/main/data/OLX_cars_dataset002.csv"
    
    # Taxa de câmbio de Rúpia Paquistanesa (PKR) para Real Brasileiro (BRL)
    EXCHANGE_RATE = 0.027  # Exemplo: 1 PKR = 0.027 BRL

    # Criação do objeto de análise de carros
    car_analysis = CarAnalysis(DATA_URL, EXCHANGE_RATE)

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

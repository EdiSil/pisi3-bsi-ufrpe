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

    def format_currency(self, value):
        """
        Formata um valor no formato monetário brasileiro (R$) sem depender da localidade do sistema.
        """
        return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

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

        # Exibindo o gráfico no Streamlit
        st.plotly_chart(fig)

        # Explicação sobre a correlação
        st.markdown("""
        **Explicando:**
        
        1° **Correlação entre "ano" e "preço":** 0.68 (Correlação positiva forte). Essa correlação sugere que veículos mais recentes (ano mais alto) tendem a ter um preço maior. O coeficiente de 0.68 indica uma relação linear positiva considerável. Isso é esperado, pois veículos novos geralmente têm maior valor de mercado em comparação aos veículos antigos.

        2° **Correlação entre "quilometragem" e "preço":** -0.19 (Correlação negativa fraca). A correlação entre a quilometragem e o preço é negativa e fraca. Isso significa que, embora haja uma tendência de que veículos com mais quilometragem tenham um preço menor, essa relação não é forte. Em outras palavras, a quilometragem impacta o preço, mas há outros fatores mais relevantes influenciando essa variável.

        3° **Correlação entre "ano" e "quilometragem":** -0.38 (Correlação negativa moderada). Essa correlação mostra que veículos mais novos tendem a ter menos quilometragem. O valor de -0.38 indica uma relação linear negativa moderada. Isso pode ser explicado pelo fato de veículos mais antigos, naturalmente, acumularem maior quilometragem com o tempo, enquanto veículos recentes ainda não tiveram tempo para percorrer grandes distâncias.

        **Conclusão:** 
        
        Veículos mais novos tendem a ter preços mais altos e menor quilometragem. A quilometragem tem uma relação negativa fraca com o preço, sugerindo que outros fatores, como o estado de conservação, ano e modelo, podem ter mais impacto no preço do que apenas a quilometragem.
        """)

    def plot_interactive_scatter(self):
        """
        Exibe um gráfico de dispersão interativo entre 'ano' e 'preco' categorizado por 'marca'.
        """
        # Adicionar coluna de preço formatado para exibir no hover
        self.df['preco_formatado'] = self.df['preco'].apply(self.format_currency)

        fig = px.scatter(
            self.df, x='ano', y='preco', color='marca',
            hover_data={'modelo': True, 'combustivel': True, 'tipo': True, 'preco_formatado': True},
            title="Preço x Ano por Marca",
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

if __name__ == "__main__":
    run_app()

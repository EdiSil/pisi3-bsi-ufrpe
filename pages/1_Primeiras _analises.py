import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Função para formatar valores como moeda brasileira
def format_brl(value):
    return f"R${value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Função para converter valores de string para float
def convert_to_float(value):
    return float(str(value).replace('R$', '').replace('.', '').replace(',', '.'))

class CarAnalysisApp:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.brand_colors = {}

    def load_data(self):
        """Carrega os dados do arquivo CSV."""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['preco'] = self.df['preco'].apply(convert_to_float)
            st.sidebar.success("DADOS CARREGADOS COM SUCESSO!")
        except Exception as e:
            st.sidebar.error(f"ERRO AO CARREGAR OS DADOS: {e}")

    def filter_top_10_brands(self):
        """Filtra as 10 marcas com mais veículos."""
        if self.df is not None:
            top_brands = self.df['marca'].value_counts().head(10).index
            self.df = self.df[self.df['marca'].isin(top_brands)]
            self.brand_colors = {brand: px.colors.qualitative.Plotly[i] for i, brand in enumerate(top_brands)}

    def show_boxplot_by_quilometragem(self):
        """Exibe um boxplot das marcas por quilometragem."""
        st.subheader("BOXPLOT: QUILOMETRAGEM POR MARCA")
        if self.df is not None:
            fig = px.box(self.df, x='marca', y='quilometragem', title='BOXPLOT DAS MARCAS POR QUILOMETRAGEM', 
                         color='marca', color_discrete_map=self.brand_colors)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

    def show_histogram_by_brand(self):
        """Exibe um histograma da quantidade de veículos por marca."""
        st.subheader("HISTOGRAMA: QUANTIDADE DE VEÍCULOS POR MARCA")
        if self.df is not None:
            vehicle_counts = self.df['marca'].value_counts().reset_index()
            vehicle_counts.columns = ['marca', 'unidades']

            fig = px.bar(vehicle_counts, x='marca', y='unidades', title='HISTOGRAMA DA QUANTIDADE DE VEÍCULOS POR MARCA', 
                         color='marca', color_discrete_map=self.brand_colors)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

    def show_bar_chart_preco_ano(self):
        """Exibe um gráfico de barras relacionando preço e ano acumulado."""
        st.subheader("GRÁFICO DE BARRAS: PREÇO TOTAL ACUMULADO POR ANO")
        if self.df is not None:
            price_per_year = self.df.groupby('ano')['preco'].sum().reset_index()

            fig = px.bar(price_per_year, x='ano', y='preco', 
                         title='RELAÇÃO ENTRE PREÇOS TOTAIS ACUMULADOS POR ANO', 
                         color='ano', color_continuous_scale='Viridis')
            fig.update_layout(
                yaxis_title="PREÇO TOTAL (R$)",
                xaxis_title="ANO",
                hovermode="x unified",
                showlegend=False
            )
            st.plotly_chart(fig)

    def show_scatter_plot(self):
        """Exibe um gráfico de dispersão interativo."""
        st.subheader("GRÁFICO DE DISPERSÃO")
        if self.df is not None:
            fig = px.scatter(self.df, x='preco', y='quilometragem', color='marca', 
                             hover_data=['ano', 'modelo', 'combustivel', 'tipo'],
                             title='GRÁFICO DE DISPERSÃO: PREÇO X QUILOMETRAGEM', 
                             color_discrete_map=self.brand_colors)
            fig.update_layout(yaxis_title="QUILOMETRAGEM (KM)", showlegend=False)
            st.plotly_chart(fig)

    def show_pie_chart_combustivel(self):
        """Exibe um gráfico de pizza da distribuição de combustíveis."""
        st.subheader("GRÁFICO DE PIZZA: DISTRIBUIÇÃO DE COMBUSTÍVEIS")
        if self.df is not None:
            fuel_counts = self.df['combustivel'].value_counts().reset_index()
            fuel_counts.columns = ['combustivel', 'unidades']

            fig = px.pie(fuel_counts, values='unidades', names='combustivel', title='DISTRIBUIÇÃO DE COMBUSTÍVEIS')
            st.plotly_chart(fig)

    def show_line_chart_quilometragem_ano(self):
        """Exibe um gráfico de linha da quilometragem média por ano."""
        st.subheader("GRÁFICO DE LINHA: QUILOMETRAGEM MÉDIA POR ANO")
        if self.df is not None:
            avg_km_per_year = self.df.groupby('ano')['quilometragem'].mean().reset_index()

            fig = px.line(avg_km_per_year, x='ano', y='quilometragem', 
                          title='EVOLUÇÃO DA QUILOMETRAGEM MÉDIA POR ANO')
            fig.update_layout(
                xaxis_title="ANO",
                yaxis_title="QUILOMETRAGEM MÉDIA (KM)"
            )
            st.plotly_chart(fig)

    def show_heatmap_correlacao(self):
        """Exibe um heatmap de correlação entre variáveis."""
        st.subheader("HEATMAP: CORRELAÇÃO ENTRE VARIÁVEIS")
        if self.df is not None:
            corr_matrix = self.df.corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect='auto', title='CORRELAÇÃO ENTRE VARIÁVEIS')
            st.plotly_chart(fig)

    def run_app(self):
        """Executa todos os métodos da aplicação."""
        st.title("ANÁLISE EXPLORATÓRIA DE VEÍCULOS")
        self.load_data()
        self.filter_top_10_brands()
        
        self.show_boxplot_by_quilometragem()
        self.show_histogram_by_brand()
        self.show_bar_chart_preco_ano()
        self.show_scatter_plot()
        self.show_pie_chart_combustivel()
        self.show_line_chart_quilometragem_ano()
        self.show_heatmap_correlacao()

# Caminho do arquivo CSV
data_path = "Datas/1_Cars_dataset_processado.csv"

# Inicializa o aplicativo
if __name__ == "__main__":
    app = CarAnalysisApp(data_path)
    app.run_app()

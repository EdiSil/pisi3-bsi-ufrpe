import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

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
        st.subheader("BOXPLOT: QUILOMETRAGEM POR MARCA")
        if self.df is not None:
            fig = px.box(self.df, x='marca', y='quilometragem', title='BOXPLOT DAS MARCAS POR QUILOMETRAGEM', 
                         color='marca', color_discrete_map=self.brand_colors)
            fig.update_layout(showlegend=False, xaxis_title='MARCA', yaxis_title='QUILOMETRAGEM (KM)')
            st.plotly_chart(fig)

    def show_histogram_by_brand(self):
        st.subheader("HISTOGRAMA: QUANTIDADE DE VEÍCULOS POR MARCA")
        if self.df is not None:
            vehicle_counts = self.df['marca'].value_counts().reset_index()
            vehicle_counts.columns = ['marca', 'unidades']

            fig = px.bar(vehicle_counts, x='marca', y='unidades', title='HISTOGRAMA DA QUANTIDADE DE VEÍCULOS POR MARCA', 
                         color='marca', color_discrete_map=self.brand_colors)
            fig.update_layout(showlegend=False, xaxis_title='MARCA', yaxis_title='UNIDADES')
            st.plotly_chart(fig)

    def show_bar_chart_preco_ano(self):
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
        st.subheader("GRÁFICO DE DISPERSÃO")
        if self.df is not None:
            fig = px.scatter(self.df, x='preco', y='quilometragem', color='marca', 
                             hover_data=['ano', 'modelo', 'combustivel', 'tipo'],
                             title='GRÁFICO DE DISPERSÃO: PREÇO X QUILOMETRAGEM', 
                             color_discrete_map=self.brand_colors)
            fig.update_layout(yaxis_title="QUILOMETRAGEM (KM)", xaxis_title="PREÇO (R$)", showlegend=False)
            st.plotly_chart(fig)

    def show_pie_chart_by_fuel(self):
        st.subheader("GRÁFICO DE PIZZA: DISTRIBUIÇÃO POR COMBUSTÍVEL")
        if self.df is not None:
            fuel_counts = self.df['combustivel'].value_counts().reset_index()
            fuel_counts.columns = ['combustivel', 'unidades']

            fig = px.pie(fuel_counts, values='unidades', names='combustivel', title='DISTRIBUIÇÃO DE VEÍCULOS POR COMBUSTÍVEL')
            st.plotly_chart(fig)

    def show_line_chart_price_over_time(self):
        st.subheader("GRÁFICO DE LINHA: PREÇO AO LONGO DOS ANOS")
        if self.df is not None:
            avg_price_per_year = self.df.groupby('ano')['preco'].mean().reset_index()

            fig = px.line(avg_price_per_year, x='ano', y='preco', title='PREÇO MÉDIO AO LONGO DOS ANOS')
            fig.update_layout(yaxis_title="PREÇO MÉDIO (R$)", xaxis_title="ANO")
            st.plotly_chart(fig)

    def show_stacked_bar_chart(self):
        st.subheader("GRÁFICO BARRAS EMPILHADAS: TIPO DE VEÍCULO POR ANO")
        if self.df is not None:
            stacked_data = self.df.groupby(['ano', 'tipo']).size().reset_index(name='contagem')
            fig = px.bar(stacked_data, x='ano', y='contagem', color='tipo', title='DISTRIBUIÇÃO DE VEÍCULOS POR TIPO E ANO')
            st.plotly_chart(fig)

    def show_heatmap(self):
        st.subheader("MAPA DE CALOR: CORRELAÇÃO ENTRE VARIÁVEIS")
        if self.df is not None:
            corr = self.df[['preco', 'quilometragem', 'ano']].corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
            st.pyplot(plt)

    def dashboard_controls(self):
        st.sidebar.title("PAINEL DE CONTROLE")
        marcas_selecionadas = st.sidebar.multiselect("SELECIONE AS MARCAS", self.df['marca'].unique(), default=self.df['marca'].unique())
        ano_min, ano_max = st.sidebar.slider("ANO DE FABRICAÇÃO", int(self.df['ano'].min()), int(self.df['ano'].max()), (int(self.df['ano'].min()), int(self.df['ano'].max()))
        )
        quilometragem_max = st.sidebar.slider("QUILOMETRAGEM MÁXIMA", 0, int(self.df['quilometragem'].max()), int(self.df['quilometragem'].max()))

        self.df = self.df[(self.df['marca'].isin(marcas_selecionadas)) & 
                          (self.df['ano'] >= ano_min) & 
                          (self.df['ano'] <= ano_max) & 
                          (self.df['quilometragem'] <= quilometragem_max)]

    def run_app(self):
        st.title("PRIMEIRAS ANÁLISES")
        self.load_data()
        self.filter_top_10_brands()
        self.dashboard_controls()

        self.show_boxplot_by_quilometragem()
        self.show_histogram_by_brand()
        self.show_bar_chart_preco_ano()
        self.show_scatter_plot()
        self.show_pie_chart_by_fuel()
        self.show_line_chart_price_over_time()
        self.show_stacked_bar_chart()
        self.show_heatmap()

if __name__ == "__main__":
    data_path = "Datas/1_Cars_processado.csv"
    app = CarAnalysisApp(data_path)
    app.run_app()


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
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['preco'] = self.df['preco'].apply(convert_to_float)
            st.success("Dados carregados com sucesso!")
        except Exception as e:
            st.error(f"Erro ao carregar os dados: {e}")

    def filter_top_10_brands(self):
        if self.df is not None:
            top_brands = self.df['marca'].value_counts().head(10).index
            self.df = self.df[self.df['marca'].isin(top_brands)]
            self.brand_colors = {brand: px.colors.qualitative.Plotly[i] for i, brand in enumerate(top_brands)}

    def show_boxplot_by_quilometragem(self):
        st.subheader("Boxplot: Quilometragem por Marca")
        if self.df is not None:
            fig = px.box(self.df, x='marca', y='quilometragem', color='marca', color_discrete_map=self.brand_colors)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

    def show_histogram_by_brand(self):
        st.subheader("Histograma: Quantidade de Veículos por Marca")
        if self.df is not None:
            vehicle_counts = self.df['marca'].value_counts().reset_index()
            vehicle_counts.columns = ['marca', 'unidades']
            fig = px.bar(vehicle_counts, x='marca', y='unidades', color='marca', color_discrete_map=self.brand_colors)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

    def show_bar_chart_preco_ano(self):
        st.subheader("Gráfico de Barras: Preço Total Acumulado por Ano")
        if self.df is not None:
            total_price_per_year = self.df.groupby('ano')['preco'].sum().reset_index()
            total_price_per_year['preco_formatado'] = total_price_per_year['preco'].apply(format_brl)
            fig = px.bar(total_price_per_year, x='ano', y='preco', title='Preço Total Acumulado por Ano',
                         color='ano', color_continuous_scale='Viridis')
            fig.update_layout(yaxis_title="Preço Total (R$)", xaxis_title="Ano", hovermode="x unified")
            fig.add_trace(
                go.Scatter(x=total_price_per_year['ano'], y=total_price_per_year['preco'],
                           mode='lines+markers', name='Tendência', line=dict(color='red'))
            )
            st.plotly_chart(fig)

    def show_scatter_plot(self):
        st.subheader("Gráfico de Dispersão: Preço x Quilometragem")
        if self.df is not None:
            fig = px.scatter(self.df, x='preco', y='quilometragem', color='marca', hover_data=['ano', 'modelo'])
            fig.update_layout(yaxis_title="Quilometragem (Km)", showlegend=False)
            st.plotly_chart(fig)

    def run_app(self):
        st.title("Análise de Dados de Veículos")
        self.load_data()
        self.filter_top_10_brands()
        self.show_boxplot_by_quilometragem()
        self.show_histogram_by_brand()
        self.show_bar_chart_preco_ano()
        self.show_scatter_plot()

# Caminho do arquivo CSV
data_path = "Datas/1_Cars_dataset_processado.csv"

if __name__ == "__main__":
    app = CarAnalysisApp(data_path)
    app.run_app()

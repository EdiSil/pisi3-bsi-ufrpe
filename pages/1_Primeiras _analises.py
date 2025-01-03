import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Função para formatar valores em R$
def format_brl(value):
    return f"R${value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Função para converter valores monetários de string para float
def convert_to_float(value):
    return float(str(value).replace('R$', '').replace('.', '').replace(',', '.'))

# Classe para Análise de Dados de Carros
class CarDataAnalysisApp:
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
            st.error(f"Erro ao carregar dados: {e}")

    def filter_top_brands(self):
        top_brands = self.df['marca'].value_counts().head(10).index
        self.df = self.df[self.df['marca'].isin(top_brands)]
        self.brand_colors = {brand: px.colors.qualitative.Plotly[i] for i, brand in enumerate(top_brands)}

    def boxplot_price_by_brand(self):
        st.subheader("Distribuição de Preços por Marca")
        fig = px.box(self.df, x='marca', y='preco', color='marca',
                     title='Distribuição de Preço por Marca',
                     color_discrete_map=self.brand_colors)
        fig.update_layout(showlegend=False, yaxis_title="Preço (R$)")
        st.plotly_chart(fig)

    def violinplot_km_by_year(self):
        st.subheader("Distribuição de Quilometragem por Ano")
        fig = px.violin(self.df, x='ano', y='quilometragem', box=True,
                        title='Distribuição de Quilometragem por Ano')
        fig.update_layout(yaxis_title="Quilometragem (Km)")
        st.plotly_chart(fig)

    def scatter_price_vs_km(self):
        st.subheader("Correlação entre Preço e Quilometragem")
        fig = px.scatter(self.df, x='quilometragem', y='preco', color='marca',
                         title='Relação entre Preço e Quilometragem',
                         color_discrete_map=self.brand_colors)
        fig.update_layout(yaxis_title="Preço (R$)", xaxis_title="Quilometragem (Km)")
        st.plotly_chart(fig)

    def histogram_vehicle_count_by_brand(self):
        st.subheader("Quantidade de Veículos por Marca")
        brand_counts = self.df['marca'].value_counts().reset_index()
        brand_counts.columns = ['marca', 'unidades']
        fig = px.bar(brand_counts, x='marca', y='unidades', color='marca',
                     title='Quantidade de Veículos por Marca',
                     color_discrete_map=self.brand_colors, text='unidades')
        fig.update_layout(showlegend=False, yaxis_title="Quantidade de Veículos")
        st.plotly_chart(fig)

    def bar_avg_price_by_year(self):
        st.subheader("Preço Médio por Ano")
        avg_price = self.df.groupby('ano')['preco'].mean().reset_index()
        avg_price['preco_formatado'] = avg_price['preco'].apply(format_brl)
        fig = px.bar(avg_price, x='ano', y='preco', text='preco_formatado',
                     title='Preço Médio por Ano', color='ano', color_continuous_scale='Viridis')
        fig.add_trace(go.Scatter(x=avg_price['ano'], y=avg_price['preco'],
                                 mode='lines+markers', name='Tendência'))
        fig.update_layout(yaxis_title="Preço Médio (R$)")
        st.plotly_chart(fig)

    def heatmap_correlation(self):
        st.subheader("Mapa de Calor de Correlação entre Variáveis")
        corr = self.df[['preco', 'quilometragem', 'ano']].corr()
        plt.figure(figsize=(10, 6))
        heatmap = sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        st.pyplot(plt)

    def run(self):
        st.title("Exploração de Dados de Veículos")
        self.load_data()
        self.filter_top_brands()
        self.boxplot_price_by_brand()
        self.violinplot_km_by_year()
        self.scatter_price_vs_km()
        self.histogram_vehicle_count_by_brand()
        self.bar_avg_price_by_year()
        self.heatmap_correlation()

# Inicializa a aplicação
if __name__ == "__main__":
    data_path = "Datas/1_Cars_dataset_processado.csv"
    app = CarDataAnalysisApp(data_path)
    app.run()

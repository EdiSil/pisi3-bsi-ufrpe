import pandas as pd
import streamlit as st
import plotly.express as px

# Função para formatar valores para Real Brasileiro

def format_to_brl(value):
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

class CarAnalysisApp:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.df_filtered = None

    def load_data(self):
        """Carrega os dados do arquivo CSV."""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df_filtered = self.df.copy()
            st.sidebar.success("Dados carregados com sucesso!")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar os dados: {e}")

    def add_filters(self):
        """Adiciona filtros interativos no painel lateral."""
        if self.df is not None:
            marcas = sorted(self.df['marca'].unique())
            modelos = sorted(self.df['modelo'].unique())

            marca_selecionada = st.sidebar.multiselect("Selecione a Marca:", marcas, default=marcas)
            modelo_selecionado = st.sidebar.multiselect("Selecione o Modelo:", modelos, default=modelos)

            ano_min, ano_max = st.sidebar.slider("Ano de Fabricação:", int(self.df['ano'].min()), int(self.df['ano'].max()), (2000, 2023))
            preco_min, preco_max = st.sidebar.slider("Faixa de Preço (R$):", int(self.df['preco'].min()), int(self.df['preco'].max()), (245000, 5000000))

            self.df_filtered = self.df[(self.df['marca'].isin(marca_selecionada)) &
                                       (self.df['modelo'].isin(modelo_selecionado)) &
                                       (self.df['ano'] >= ano_min) & (self.df['ano'] <= ano_max) &
                                       (self.df['preco'] >= preco_min) & (self.df['preco'] <= preco_max)]

    def show_price_distribution(self):
        """Distribuição de Preços por Marca e Modelo."""
        fig = px.box(self.df_filtered, x='marca', y='preco', color='marca',
                     title='Distribuição de Preços por Marca',
                     labels={'marca': 'Marca', 'preco': 'Preço (R$)'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)

    def show_kilometer_distribution(self):
        """Distribuição de Quilometragem por Marca."""
        fig = px.histogram(self.df_filtered, x='quilometragem', color='marca', nbins=50,
                           title='Distribuição de Quilometragem dos Veículos',
                           labels={'quilometragem': 'Quilometragem (km)', 'marca': 'Marca'})
        st.plotly_chart(fig)

    def show_avg_price_by_model(self):
        """Preço Médio por Modelo."""
        avg_price = self.df_filtered.groupby(['modelo', 'marca'])['preco'].mean().reset_index().sort_values(by='preco', ascending=False)
        fig = px.bar(avg_price, x='modelo', y='preco', color='marca', title='Preço Médio por Modelo',
                     labels={'modelo': 'Modelo', 'preco': 'Preço Médio (R$)'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)

    def show_density_contour(self):
        """Mapa de Densidade entre Preço e Ano de Fabricação."""
        fig = px.density_contour(self.df_filtered, x='ano', y='preco',
                                 title='Densidade de Preço por Ano',
                                 labels={'ano': 'Ano de Fabricação', 'preco': 'Preço (R$)'})
        st.plotly_chart(fig)

    def run_app(self):
        st.title("Análise Exploratória de Veículos")
        self.load_data()
        self.add_filters()
        self.show_price_distribution()
        self.show_kilometer_distribution()
        self.show_avg_price_by_model()
        self.show_density_contour()

if __name__ == "__main__":
    data_path = "/mnt/data/1_Cars_dataset_processado2.csv"
    app = CarAnalysisApp(data_path)
    app.run_app()


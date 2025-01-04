import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Função para converter valores de string para float
def convert_to_float(value):
    return float(str(value).replace('R$', '').replace('.', '').replace(',', '.'))

class CarAnalysisApp:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.df_filtered = None  # DataFrame filtrado pelos anos

    def load_data(self):
        """Carrega os dados do arquivo CSV."""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['preco'] = self.df['preco'].apply(convert_to_float)
            self.df_filtered = self.df.copy()  # Inicializa o DataFrame filtrado
            st.sidebar.success("Dados carregados com sucesso!")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar os dados: {e}")

    def add_year_filter(self):
        """Adiciona um filtro de ano ao painel lateral e retorna o intervalo selecionado."""
        if self.df is not None:
            anos = sorted(self.df['ano'].unique())
            ano_min, ano_max = st.sidebar.slider(
                "SELECIONE O INTERVALO DE ANOS:", 
                min_value=int(min(anos)), 
                max_value=int(max(anos)), 
                value=(int(min(anos)), int(max(anos)))
            )
            self.df_filtered = self.df[(self.df['ano'] >= ano_min) & (self.df['ano'] <= ano_max)]
            return ano_min, ano_max
        return None, None

    def show_price_distribution_by_brand(self):
        if self.df_filtered is not None:
            fig = px.scatter(
                self.df_filtered, x='marca', y='preco', 
                color='marca',
                title='DISTRIBUIÇÃO DE PREÇOS POR MARCA',
                labels={'marca': 'MARCA', 'preco': 'PREÇO (R$)'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

    def show_price_trends_over_years(self):
        if self.df_filtered is not None:
            avg_price_by_year = self.df_filtered.groupby('ano')['preco'].mean().reset_index()
            fig = px.line(
                avg_price_by_year, x='ano', y='preco', 
                title='TENDÊNCIA DE PREÇOS AO LONGO DOS ANOS',
                labels={'ano': 'ANO', 'preco': 'PREÇO MÉDIO (R$)'}
            )
            st.plotly_chart(fig)

    def show_price_by_fuel_type(self):
        if self.df_filtered is not None:
            avg_price_by_fuel = self.df_filtered.groupby('combustivel')['preco'].mean().reset_index()
            fig = px.bar(
                avg_price_by_fuel, x='combustivel', y='preco', 
                color='combustivel',
                title='PREÇOS MÉDIOS POR COMBUSTÍVEL',
                labels={'combustivel': 'COMBUSTÍVEL', 'preco': 'PREÇO MÉDIO (R$)'}
            )
            st.plotly_chart(fig)

    def show_scatter_price_vs_km(self):
        if self.df_filtered is not None:
            fig = px.scatter(
                self.df_filtered, x='quilometragem', y='preco', 
                color='marca',
                title='DISPERSÃO: PREÇO X QUILOMETRAGEM',
                labels={'quilometragem': 'QUILOMETRAGEM (KM)', 'preco': 'PREÇO (R$)'}
            )
            st.plotly_chart(fig)

    def show_histogram_by_year(self):
        if self.df_filtered is not None:
            fig = px.histogram(
                self.df_filtered, x='ano', 
                title='DISTRIBUIÇÃO DE VEÍCULOS POR ANO',
                labels={'ano': 'ANO'}
            )
            st.plotly_chart(fig)

    def show_pie_chart_by_brand(self):
        if self.df_filtered is not None:
            brand_counts = self.df_filtered['marca'].value_counts().reset_index()
            brand_counts.columns = ['marca', 'unidades']
            fig = px.pie(
                brand_counts, values='unidades', names='marca',
                title='DISTRIBUIÇÃO DE VEÍCULOS POR MARCA'
            )
            st.plotly_chart(fig)

    def show_violin_plot(self):
        if self.df_filtered is not None:
            fig = px.violin(
                self.df_filtered, y='preco', x='marca',
                box=True, points='all',
                title='DISTRIBUIÇÃO DE PREÇOS POR MARCA (VIOLINO)',
                labels={'marca': 'MARCA', 'preco': 'PREÇO (R$)'}
            )
            st.plotly_chart(fig)

    def run_app(self):
        st.title("Análise Exploratória de Veículos")
        self.load_data()
        self.add_year_filter()
        self.show_price_distribution_by_brand()
        self.show_price_trends_over_years()
        self.show_price_by_fuel_type()
        self.show_scatter_price_vs_km()
        self.show_histogram_by_year()
        self.show_pie_chart_by_brand()
        self.show_violin_plot()

# Caminho do arquivo CSV
data_path = "Datas/1_Cars_dataset_processado.csv"

if __name__ == "__main__":
    app = CarAnalysisApp(data_path)
    app.run_app()

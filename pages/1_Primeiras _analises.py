import pandas as pd
import streamlit as st
import plotly.express as px

# Classe principal da aplicação
class CarAnalysisApp:
    def __init__(self, data_path, exchange_rate=6.1651):
        self.data_path = data_path
        self.df = None
        self.brand_colors = None
        self.selected_years = None
        self.selected_fuel = None
        self.exchange_rate = exchange_rate

    def load_data(self):
        """Carrega os dados do CSV."""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df.columns = [col.lower().strip() for col in self.df.columns]  # Normaliza nomes das colunas
            st.success("Dados carregados com sucesso!")
            self.convert_price_to_real()
        except Exception as e:
            st.error(f"Erro ao carregar os dados: {e}")

    def convert_price_to_real(self):
        """Converte preços para reais."""
        if 'preco' in self.df.columns:
            self.df['preco'] = self.df['preco'] * self.exchange_rate
        else:
            st.warning("Coluna 'preco' não encontrada!")

    def filter_top_10_brands(self):
        """Filtra as 10 marcas mais frequentes."""
        if self.df is not None and 'marca' in self.df.columns:
            top_brands = self.df['marca'].value_counts().head(10).index
            self.df = self.df[self.df['marca'].isin(top_brands)]
            self.brand_colors = px.colors.qualitative.Set2[:10]
        else:
            st.warning("Coluna 'marca' não encontrada nos dados!")

    def show_boxplot_by_quilometragem(self):
        """Exibe boxplot de quilometragem por marca."""
        st.subheader("Boxplot: Quilometragem por Marca")
        if self.df is not None and 'quilometragem' in self.df.columns:
            fig = px.box(self.df, x='marca', y='quilometragem', color='marca', 
                         color_discrete_map={brand: color for brand, color in zip(self.df['marca'].unique(), self.brand_colors)})
            st.plotly_chart(fig)
        else:
            st.warning("Coluna 'quilometragem' não encontrada nos dados!")

    def show_histogram_by_brand(self):
        """Exibe histograma da quantidade de veículos por marca."""
        st.subheader("Histograma: Quantidade de Veículos por Marca")
        if self.df is not None and 'marca' in self.df.columns:
            vehicle_counts = self.df['marca'].value_counts().reset_index()
            vehicle_counts.columns = ['marca', 'unidades']
            fig = px.bar(vehicle_counts, x='marca', y='unidades', color='marca',
                         color_discrete_map={brand: color for brand, color in zip(vehicle_counts['marca'], self.brand_colors)})
            st.plotly_chart(fig)
        else:
            st.warning("Coluna 'marca' não encontrada nos dados!")

    def show_bar_chart_preco_ano(self):
        """Exibe gráfico de barras relacionando preço e ano."""
        st.subheader("Gráfico de Barras: Preço por Ano")
        if self.df is not None and 'ano' in self.df.columns and 'preco' in self.df.columns:
            self.selected_years = st.slider(
                "Selecione o intervalo de anos:",
                min_value=int(self.df['ano'].min()),
                max_value=int(self.df['ano'].max()),
                value=(int(self.df['ano'].min()), int(self.df['ano'].max()))
            )
            filtered_df = self.df[(self.df['ano'] >= self.selected_years[0]) & (self.df['ano'] <= self.selected_years[1])]
            fig = px.bar(filtered_df, x='ano', y='preco', color='marca', 
                         color_discrete_map={brand: color for brand, color in zip(self.df['marca'].unique(), self.brand_colors)})
            st.plotly_chart(fig)
        else:
            st.warning("Colunas 'ano' ou 'preco' não encontradas nos dados!")

    def show_scatter_plot(self):
        """Exibe gráfico de dispersão interativo."""
        st.subheader("Gráfico de Dispersão Interativo")
        if self.df is not None and 'quilometragem' in self.df.columns and 'preco' in self.df.columns:
            self.selected_fuel = st.selectbox(
                "Selecione o tipo de combustível:",
                options=self.df['combustivel'].unique()
            )
            filtered_df = self.df[self.df['combustivel'] == self.selected_fuel]
            fig = px.scatter(filtered_df, x='preco', y='quilometragem', color='marca', 
                             hover_data=['ano', 'modelo', 'combustivel', 'tipo'])
            st.plotly_chart(fig)
        else:
            st.warning("Colunas necessárias para o gráfico de dispersão não encontradas!")

    def run_app(self):
        """Executa a aplicação."""
        st.title("Análise de Veículos")
        self.load_data()
        self.filter_top_10_brands()

        self.show_boxplot_by_quilometragem()
        self.show_histogram_by_brand()
        self.show_bar_chart_preco_ano()
        self.show_scatter_plot()

# Caminho do arquivo CSV
data_path = "Datas/1_Cars_processado.csv"

# Inicializa e executa o aplicativo
if __name__ == "__main__":
    app = CarAnalysisApp(data_path)
    app.run_app()

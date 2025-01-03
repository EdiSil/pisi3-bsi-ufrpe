import pandas as pd
import streamlit as st
import plotly.express as px

# Função para converter valores de string para float
def convert_to_float(value):
    return float(str(value).replace('R$', '').replace('.', '').replace(',', '.'))

class CarAnalysisApp:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        """Carrega os dados do arquivo CSV."""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['preco'] = self.df['preco'].apply(convert_to_float)
            st.sidebar.success("Dados carregados com sucesso!")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar os dados: {e}")

    def add_year_filter(self):
        """Adiciona um filtro de ano ao painel lateral."""
        if self.df is not None:
            anos = sorted(self.df['ano'].unique())
            ano_min, ano_max = st.sidebar.slider(
                "Selecione o intervalo de anos:", 
                min_value=int(min(anos)), 
                max_value=int(max(anos)), 
                value=(int(min(anos)), int(max(anos)))
            )
            self.df = self.df[(self.df['ano'] >= ano_min) & (self.df['ano'] <= ano_max)]

    def show_price_vs_mileage_scatter(self):
        """Gráfico de dispersão entre preço e quilometragem."""
        if self.df is not None:
            fig = px.scatter(
                self.df, x='quilometragem', y='preco', 
                color='marca', hover_data=['ano', 'modelo', 'combustivel'],
                title='Dispersão: Preço vs Quilometragem',
                labels={'quilometragem': 'Quilometragem (km)', 'preco': 'Preço (R$)'}
            )
            st.plotly_chart(fig)

    def show_price_distribution_by_brand(self):
        """Distribuição de preços por marca."""
        if self.df is not None:
            fig = px.box(
                self.df, x='marca', y='preco', 
                color='marca',
                title='Distribuição de Preços por Marca',
                labels={'marca': 'Marca', 'preco': 'Preço (R$)'}
            )
            st.plotly_chart(fig)

    def show_price_trends_over_years(self):
        """Tendências de preços médios ao longo dos anos."""
        if self.df is not None:
            avg_price_by_year = self.df.groupby('ano')['preco'].mean().reset_index()
            fig = px.line(
                avg_price_by_year, x='ano', y='preco', 
                title='Tendência de Preços Médios ao Longo dos Anos',
                labels={'ano': 'Ano', 'preco': 'Preço Médio (R$)'}
            )
            st.plotly_chart(fig)

    def show_price_by_fuel_type(self):
        """Preços médios por tipo de combustível."""
        if self.df is not None:
            avg_price_by_fuel = self.df.groupby('combustivel')['preco'].mean().reset_index()
            fig = px.bar(
                avg_price_by_fuel, x='combustivel', y='preco', 
                color='combustivel',
                title='Preços Médios por Tipo de Combustível',
                labels={'combustivel': 'Combustível', 'preco': 'Preço Médio (R$)'}
            )
            st.plotly_chart(fig)

    def show_price_by_transmission_type(self):
        """Preços médios por tipo de transmissão."""
        if self.df is not None:
            avg_price_by_transmission = self.df.groupby('tipo')['preco'].mean().reset_index()
            fig = px.bar(
                avg_price_by_transmission, x='tipo', y='preco', 
                color='tipo',
                title='Preços Médios por Tipo de Transmissão',
                labels={'tipo': 'Tipo de Transmissão', 'preco': 'Preço Médio (R$)'}
            )
            st.plotly_chart(fig)

    def show_correlation_matrix(self):
        """Exibe uma matriz de correlação interativa."""
        if self.df is not None:
            correlation = self.df[['preco', 'quilometragem', 'ano']].corr()
            fig = px.imshow(
                correlation, text_auto=True, title="Matriz de Correlação",
                labels={'color': 'Correlação'}
            )
            st.plotly_chart(fig)

    def show_price_by_category(self):
        """Preços médios por categorias adicionais."""
        if self.df is not None:
            if 'categoria' in self.df.columns:
                avg_price_by_category = self.df.groupby('categoria')['preco'].mean().reset_index()
                fig = px.bar(
                    avg_price_by_category, x='categoria', y='preco', 
                    color='categoria',
                    title='Preços Médios por Categoria',
                    labels={'categoria': 'Categoria', 'preco': 'Preço Médio (R$)'}
                )
                st.plotly_chart(fig)
            else:
                st.warning("A coluna 'categoria' não está presente no conjunto de dados.")

    def run_app(self):
        """Executa todos os métodos da aplicação."""
        st.title("Análise Exploratória de Carros Usados")
        self.load_data()
        self.add_year_filter()
        self.show_price_vs_mileage_scatter()
        self.show_price_distribution_by_brand()
        self.show_price_trends_over_years()
        self.show_price_by_fuel_type()
        self.show_price_by_transmission_type()
        self.show_correlation_matrix()
        self.show_price_by_category()

# Caminho do arquivo CSV
data_path = "Datas/1_Cars_dataset_processado.csv"

# Inicializa o aplicativo
if __name__ == "__main__":
    app = CarAnalysisApp(data_path)
    app.run_app()

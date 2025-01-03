import pandas as pd
import streamlit as st
import plotly.express as px

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

    def load_data(self):
        """Carrega os dados do arquivo CSV."""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['preco'] = self.df['preco'].apply(convert_to_float)
            st.sidebar.success("Dados carregados com sucesso!")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar os dados: {e}")

    def show_bar_chart_preco_ano(self):
        """Exibe um gráfico de barras relacionando preço e ano acumulado."""
        st.subheader("Gráfico de Barras: Preço Total Acumulado por Ano")
        if self.df is not None:
            price_per_year = self.df.groupby('ano')['preco'].sum().reset_index()

            fig = px.bar(price_per_year, x='ano', y='preco', 
                         title='Relação entre Preços Totais Acumulados por Ano', 
                         color='ano', color_continuous_scale='Viridis')
            fig.update_layout(
                yaxis_title="Preço Total (R$)",
                xaxis_title="Ano",
                hovermode="x unified",
                showlegend=False
            )
            st.plotly_chart(fig)

    def run_app(self):
        """Executa o método da aplicação."""
        st.title("Análise Exploratoria de Veículos")
        self.load_data()
        self.show_bar_chart_preco_ano()

# Caminho do arquivo CSV
data_path = "Datas/1_Cars_dataset_processado.csv"

# Inicializa o aplicativo
if __name__ == "__main__":
    app = CarAnalysisApp(data_path)
    app.run_app()

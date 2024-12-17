import pandas as pd
import streamlit as st
import plotly.express as px

# Classe principal da aplicacao
class CarAnalysisApp:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        """Carrega os dados a partir do caminho especificado."""
        try:
            self.df = pd.read_csv(self.data_path)
        except Exception as e:
            st.error(f"Erro ao carregar os dados: {e}")

    def clean_data(self):
        """Limpa e ajusta os dados principais."""
        if self.df is not None:
            try:
                # Remove espacos extras nos nomes das colunas
                self.df.columns = self.df.columns.str.strip().str.lower()
                
                # Converte colunas relevantes para numerico
                self.df['ano'] = pd.to_numeric(self.df['ano'], errors='coerce')
                self.df['preco'] = pd.to_numeric(self.df['preco'], errors='coerce')
                self.df['quilometragem'] = pd.to_numeric(self.df['quilometragem'], errors='coerce')
                
                # Remove linhas com valores nulos nas colunas principais
                self.df.dropna(subset=['ano', 'preco', 'quilometragem', 'marca'], inplace=True)
                
                # Remove anos ou preços inválidos (ex: preço negativo ou ano absurdo)
                self.df = self.df[(self.df['ano'] > 1900) & (self.df['ano'] <= 2024)]
                self.df = self.df[self.df['preco'] > 0]
            except Exception as e:
                st.error(f"Erro ao limpar os dados: {e}")

    def filter_top_10_brands(self):
        """Filtra as 10 marcas com mais ocorrências no dataset."""
        if self.df is not None:
            top_brands = self.df['marca'].value_counts().head(10).index
            self.df = self.df[self.df['marca'].isin(top_brands)]
        else:
            st.warning("Nenhum dado carregado ainda!")

    def show_boxplot_by_quilometragem(self):
        """Exibe um boxplot das marcas por quilometragem."""
        st.subheader("Boxplot: Quilometragem por Marca")
        if self.df is not None:
            fig = px.box(
                self.df, 
                x='marca', 
                y='quilometragem', 
                title='Boxplot das Marcas por Quilometragem',
                template='plotly_white'  # Fundo branco
            )
            st.plotly_chart(fig)
        else:
            st.warning("Dados não disponíveis para exibição.")

    def show_histogram_by_brand(self):
        """Exibe um histograma da quantidade de veículos por marca."""
        st.subheader("Histograma: Quantidade de Veículos por Marca")
        if self.df is not None:
            fig = px.histogram(
                self.df, 
                x='marca', 
                title='Histograma da Quantidade de Veículos por Marca',
                color_discrete_sequence=px.colors.sequential.Magenta  # Paleta do pink ao roxo
            )
            st.plotly_chart(fig)
        else:
            st.warning("Dados não disponíveis para exibição.")

    def show_bar_chart_preco_ano(self):
        """Exibe um gráfico de barras relacionando preço e ano."""
        st.subheader("Gráfico de Barras: Preço por Ano")
        if self.df is not None:
            fig = px.bar(
                self.df, 
                x='ano', 
                y='preco', 
                color='marca', 
                title='Relação entre Preço e Ano',
                color_discrete_sequence=['#800080', '#FF69B4']  # Roxo e Pink
            )
            st.plotly_chart(fig)
        else:
            st.warning("Dados não disponíveis para exibição.")

    def show_scatter_plot(self):
        """Exibe um gráfico de dispersão interativo."""
        st.subheader("Gráfico de Dispersão Interativo")
        if self.df is not None:
            fig = px.scatter(
                self.df, 
                x='preco', 
                y='quilometragem', 
                color='marca', 
                hover_data=['ano', 'modelo', 'combustivel', 'tipo'],
                title='Gráfico de Dispersão: Preço x Quilometragem',
                color_discrete_sequence=['#800080', '#FF69B4']  # Roxo e Pink
            )
            st.plotly_chart(fig)
        else:
            st.warning("Dados não disponíveis para exibição.")

    def run_app(self):
        """Executa todos os métodos da aplicação."""
        st.title("Primeiras Análises")
        self.load_data()
        self.clean_data()
        self.filter_top_10_brands()

        # Exibindo gráficos
        self.show_boxplot_by_quilometragem()
        self.show_histogram_by_brand()
        self.show_bar_chart_preco_ano()
        self.show_scatter_plot()

# Caminho do arquivo CSV
data_path = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_dataset002.csv"

# Inicializa o aplicativo
if __name__ == "__main__":
    app = CarAnalysisApp(data_path)
    app.run_app()

import pandas as pd
import streamlit as st
import plotly.express as px

# Função para formatar valores como moeda brasileira
def format_brl(value):
    return f"R${value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Classe principal da aplicação
class CarAnalysisApp:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.brand_colors = None

    def load_data(self):
        """Carrega os dados a partir do caminho especificado."""
        try:
            self.df = pd.read_csv(self.data_path)
            st.success("Dados carregados com sucesso!")
        except Exception as e:
            st.error(f"Erro ao carregar os dados: {e}")

    def filter_top_10_brands(self):
        """Filtra as 10 marcas com mais ocorrências no dataset."""
        if self.df is not None:
            top_brands = self.df['marca'].value_counts().head(10).index
            self.df = self.df[self.df['marca'].isin(top_brands)]
            # Criando uma paleta de cores baseada nas 10 principais marcas
            self.brand_colors = px.colors.qualitative.Set2[:10]
        else:
            st.warning("Nenhum dado carregado ainda!")

    def show_boxplot_by_quilometragem(self):
        """Exibe um boxplot das marcas por quilometragem."""
        st.subheader("Boxplot: Quilometragem por Marca")
        if self.df is not None:
            quilometragem_ticks = [100000, 200000, 300000, 400000, 500000]
            quilometragem_ticks_labels = ['100 Km', '200 Km', '300 Km', '400 Km', '500 Km']

            fig = px.box(self.df, x='marca', y='quilometragem', title='Boxplot das Marcas por Quilometragem', 
                         color='marca', color_discrete_map={brand: color for brand, color in zip(self.df['marca'].unique(), self.brand_colors)},
                         hover_data={'marca': True, 'quilometragem': True})

            fig.update_layout(
                yaxis_title="Quilometragem",
                yaxis=dict(
                    tickvals=quilometragem_ticks,
                    ticktext=quilometragem_ticks_labels
                )
            )
            st.plotly_chart(fig)
        else:
            st.warning("Dados não disponíveis para exibição.")

    def show_histogram_by_brand(self):
        """Exibe um histograma da quantidade de veículos por marca."""
        st.subheader("Histograma: Quantidade de Veículos por Marca")
        if self.df is not None:
            vehicle_counts = self.df['marca'].value_counts().reset_index()
            vehicle_counts.columns = ['marca', 'unidades']

            fig = px.bar(vehicle_counts, x='marca', y='unidades', title='Histograma da Quantidade de Veículos por Marca', 
                         color='marca', color_discrete_map={brand: color for brand, color in zip(vehicle_counts['marca'], self.brand_colors)},
                         text='unidades')

            fig.update_traces(hovertemplate='Marca: %{x}<br>Unidades: %{y}')
            fig.update_layout(yaxis_title="Unidades")

            st.plotly_chart(fig)
        else:
            st.warning("Dados não disponíveis para exibição.")

    def show_bar_chart_preco_ano(self):
        """Exibe um gráfico de barras relacionando preço e ano."""
        st.subheader("Gráfico de Barras: Preço por Ano")
        if self.df is not None:
            # Agrupar os dados por ano e calcular a média do preço
            avg_price_per_year = self.df.groupby('ano')['preco'].mean().reset_index()

            # Formatando os valores de preço no formato de moeda brasileira manualmente
            avg_price_per_year['preco'] = avg_price_per_year['preco'].apply(lambda x: format_brl(x))

            # Exibindo o gráfico de barras
            fig = px.bar(avg_price_per_year, x='ano', y='preco', title='Relação entre Preço e Ano', 
                         color='ano', text='preco',
                         color_continuous_scale='Viridis')

            # Atualizando o título do eixo Y e as informações no hover
            fig.update_layout(
                yaxis_title="Preço Médio (R$)",
                xaxis_title="Ano",
                hovermode="x unified"  # Agrupar os dados do hover por ano
            )

            # Adicionando as informações de preço no topo das barras
            fig.update_traces(texttemplate='%{text}', textposition='outside', insidetextanchor='start')

            # Adicionando uma linha de tendência para mostrar a evolução do preço
            fig.add_trace(
                go.Scatter(x=avg_price_per_year['ano'], y=avg_price_per_year['preco'].apply(lambda x: float(x.replace('R$', '').replace('.', '').replace(',', '.'))), 
                           mode='lines+markers', name='Tendência', line=dict(color='red'))
            )

            st.plotly_chart(fig)
        else:
            st.warning("Dados não disponíveis para exibição.")

    def show_scatter_plot(self):
        """Exibe um gráfico de dispersão interativo."""
        st.subheader("Gráfico de Dispersão Interativo")
        if self.df is not None:
            fig = px.scatter(self.df, x='preco', y='quilometragem', color='marca', 
                             hover_data=['ano', 'modelo', 'combustivel', 'tipo'],
                             title='Gráfico de Dispersão: Preço x Quilometragem', 
                             color_discrete_map={brand: color for brand, color in zip(self.df['marca'].unique(), self.brand_colors)})
            st.plotly_chart(fig)
        else:
            st.warning("Dados não disponíveis para exibição.")

    def run_app(self):
        """Executa todos os métodos da aplicação."""
        st.title("Primeiras Análises")
        self.load_data()
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

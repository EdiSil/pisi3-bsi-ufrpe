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
        self.exchange_rate = exchange_rate  # Taxa de câmbio do dólar para real

    def load_data(self):
        """Carrega os dados a partir do caminho especificado."""
        try:
            self.df = pd.read_csv(self.data_path)
            self.rename_columns()
            st.success("Dados carregados com sucesso!")
            self.convert_price_to_real()  # Converte o preço assim que os dados são carregados
            self.filter_years(2000, 2023)  # Filtra os anos entre 2000 e 2023
        except Exception as e:
            st.error(f"Erro ao carregar os dados: {e}")

    def rename_columns(self):
        """Renomeia as colunas do DataFrame para os nomes especificados."""
        column_mapping = {
            "Car Name": "nome",
            "Make": "marca",
            "Model": "modelo",
            "KM's driven": "quilometragem (Km)",
            "Prince": "preco",
            "Fuel_Diesel": "combustivel_Diesel",
            "Fuel_Hybrid": "combustivel_Hybrido",
            "Fuel_Petrol": "combustivel_Gasolina",
            "Transmission_Manual": "Transmission_Manual",
            "Car Age": "Ano"
        }
        self.df.rename(columns=column_mapping, inplace=True)

    def convert_price_to_real(self):
        """Converte os preços de dólares para reais, multiplicando pela taxa de câmbio."""
        if 'preco' in self.df.columns:
            self.df['preco'] = self.df['preco'] * self.exchange_rate
        else:
            st.warning("Coluna 'preco' não encontrada!")

    def filter_years(self, start_year, end_year):
        """Filtra o DataFrame para incluir apenas os anos no intervalo especificado."""
        if 'Ano' in self.df.columns:
            self.df = self.df[(self.df['Ano'] >= start_year) & (self.df['Ano'] <= end_year)]
        else:
            st.warning("Coluna 'Ano' não encontrada!")

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
            fig = px.box(self.df, x='marca', y='quilometragem (Km)', title='Boxplot das Marcas por Quilometragem', 
                         color='marca', color_discrete_map={brand: color for brand, color in zip(self.df['marca'].unique(), self.brand_colors)})
            fig.update_layout(yaxis_title="Quilometragem (Km)", showlegend=False)
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
            fig.update_layout(yaxis_title="Unidades", showlegend=False)
            st.plotly_chart(fig)
        else:
            st.warning("Dados não disponíveis para exibição.")

    def show_bar_chart_preco_ano(self):
        """Exibe um gráfico de barras relacionando preço e ano."""
        st.subheader("Gráfico de Barras: Preço por Ano")
        if self.df is not None:
            self.selected_years = st.slider(
                "Selecione o intervalo de anos:",
                min_value=2000,
                max_value=2023,
                value=(2000, 2023),
                step=1
            )
            filtered_df = self.df[(self.df['Ano'] >= self.selected_years[0]) & (self.df['Ano'] <= self.selected_years[1])]

            if filtered_df.empty:
                st.warning("Nenhum dado disponível para o intervalo selecionado.")
                return

            fig = px.bar(filtered_df, x='Ano', y='preco', color='marca', title='Relação entre Preço e Ano', 
                         color_discrete_map={brand: color for brand, color in zip(self.df['marca'].unique(), self.brand_colors)})
            fig.update_layout(yaxis_title="Preço (R$)", xaxis_title="Ano", showlegend=False)
            st.plotly_chart(fig)
        else:
            st.warning("Dados não disponíveis para exibição.")

    def show_scatter_plot(self):
        """Exibe um gráfico de dispersão interativo."""
        st.subheader("Gráfico de Dispersão Interativo")
        if self.df is not None:
            self.selected_fuel = st.selectbox(
                "Selecione o tipo de combustível:",
                options=self.df['combustivel_Gasolina'].unique(),
                index=0
            )

            filtered_df = self.df[
                (self.df['combustivel_Gasolina'] == self.selected_fuel) & 
                (self.df['Ano'] >= self.selected_years[0]) &
                (self.df['Ano'] <= self.selected_years[1])
            ]

            if filtered_df.empty:
                st.warning("Nenhum dado disponível para o tipo de combustível selecionado.")
                return

            fig = px.scatter(filtered_df, x='preco', y='quilometragem (Km)', color='marca', 
                             hover_data=['Ano', 'modelo', 'combustivel_Gasolina', 'Transmission_Manual'],
                             title='Gráfico de Dispersão: Preço x Quilometragem', 
                             color_discrete_map={brand: color for brand, color in zip(self.df['marca'].unique(), self.brand_colors)})
            fig.update_layout(xaxis_title="Preço (R$)", yaxis_title="Quilometragem (Km)", showlegend=False)
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
data_path = "Datas/1_Cars_processado.csv"

# Inicializa o aplicativo
if __name__ == "__main__":
    app = CarAnalysisApp(data_path, exchange_rate=6.1651)
    app.run_app()

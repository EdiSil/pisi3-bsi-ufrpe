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
            st.success("Dados carregados com sucesso!")
            self.rename_columns()
            self.filter_years()  # Filtra os anos de 2000 a 2023
            self.convert_price_to_real()
        except Exception as e:
            st.error(f"Erro ao carregar os dados: {e}")

    def rename_columns(self):
        """Renomeia as colunas do DataFrame para os nomes especificados."""
        column_mapping = {
            'Car Name': 'nome',
            'Make': 'marca',
            'Model': 'modelo',
            "KM's driven": 'quilometragem (Km)',
            'Prince': 'preco',
            'Fuel_Diesel': 'conbustivel_Diesel',
            'Fuel_Hybrid': 'conbustivel_Hybrido',
            'Fuel_Petrol': 'combustivel_Gasolina',
            'Transmission_Manual': 'Transmission_Manual',
            'Car Age': 'Ano'
        }
        self.df.rename(columns=column_mapping, inplace=True)

    def filter_years(self):
        """Filtra os dados para incluir apenas os anos de 2000 a 2023."""
        if 'Ano' in self.df.columns:
            self.df = self.df[(self.df['Ano'] >= 2000) & (self.df['Ano'] <= 2023)]
        else:
            st.warning("Coluna 'Ano' não encontrada!")

    def convert_price_to_real(self):
        """Converte os preços de dólares para reais, multiplicando pela taxa de câmbio."""
        if 'preco' in self.df.columns:
            self.df['preco'] = self.df['preco'] * self.exchange_rate
        else:
            st.warning("Coluna 'preco' não encontrada!")

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
            # Criando a lista de valores personalizados para o eixo Y
            quilometragem_ticks = [100000, 200000, 300000, 400000, 500000]
            quilometragem_ticks_labels = ['100 Km', '200 Km', '300 Km', '400 Km', '500 Km']

            # Exibindo o Boxplot com detalhes de marca e quilometragem
            fig = px.box(self.df, x='marca', y='quilometragem (Km)', title='Boxplot das Marcas por Quilometragem', 
                         color='marca', color_discrete_map={brand: color for brand, color in zip(self.df['marca'].unique(), self.brand_colors)})

            # Atualizando o título do eixo Y
            fig.update_layout(
                yaxis_title="Quilometragem",
                yaxis=dict(
                    tickvals=quilometragem_ticks,  # Definindo os valores
                    ticktext=quilometragem_ticks_labels  # Definindo os rótulos correspondentes
                ),
                showlegend=False  # Removendo a legenda
            )
            st.plotly_chart(fig)
        else:
            st.warning("Dados não disponíveis para exibição.")

    def show_histogram_by_brand(self):
        """Exibe um histograma da quantidade de veículos por marca."""
        st.subheader("Histograma: Quantidade de Veículos por Marca")
        if self.df is not None:
            # Contando o número de veículos por marca
            vehicle_counts = self.df['marca'].value_counts().reset_index()
            vehicle_counts.columns = ['marca', 'unidades']

            # Exibindo o histograma
            fig = px.bar(vehicle_counts, x='marca', y='unidades', title='Histograma da Quantidade de Veículos por Marca', 
                         color='marca', color_discrete_map={brand: color for brand, color in zip(vehicle_counts['marca'], self.brand_colors)},
                         text='unidades')  # Mostrando o número de unidades nas barras

            # Atualizando os detalhes no hover
            fig.update_traces(hovertemplate='Marca: %{x}<br>Unidades: %{y}')

            # Atualizando o título do eixo Y
            fig.update_layout(yaxis_title="Unidades", showlegend=False)  # Removendo a legenda

            st.plotly_chart(fig)
        else:
            st.warning("Dados não disponíveis para exibição.")

    def show_bar_chart_preco_ano(self):
        """Exibe um gráfico de barras relacionando preço e ano."""
        st.subheader("Gráfico de Barras: Preço por Ano")
        if self.df is not None:
            # Adicionando interatividade para seleção de uma faixa de anos
            self.selected_years = st.slider(
                "Selecione o intervalo de anos:",
                min_value=2000,
                max_value=2023,
                value=(2000, 2023),
                step=1
            )
            filtered_df = self.df[(self.df['Ano'] >= self.selected_years[0]) & (self.df['Ano'] <= self.selected_years[1])]

            fig = px.bar(filtered_df, x='Ano', y='preco', color='marca', title='Relação entre Preço e Ano', 
                         color_discrete_map={brand: color for brand, color in zip(self.df['marca'].unique(), self.brand_colors)})

            # Atualizando o eixo Y para formatar como "R$"
            fig.update_layout(
                yaxis_title="Preço (R$)", 
                yaxis_tickprefix="R$ ",
                showlegend=False
            )

            st.plotly_chart(fig)
        else:
            st.warning("Dados não disponíveis para exibição.")

    def show_scatter_plot(self):
        """Exibe um gráfico de dispersão interativo."""
        st.subheader("Gráfico de Dispersão Interativo")
        if self.df is not None:
            # Adicionando interatividade para selecionar o tipo de combustível
            self.selected_fuel = st.selectbox(
                "Selecione o tipo de combustível:",
                options=self.df['combustivel_Gasolina'].unique(),
                index=0
            )

            # Filtrando o DataFrame com base nas interações (ano e combustível)
            filtered_df = self.df[
                (self.df['combustivel_Gasolina'] == self.selected_fuel) & 
                (self.df['Ano'] >= self.selected_years[0]) &
                (self.df['Ano'] <= self.selected_years[1])
            ]

            fig = px.scatter(filtered_df, x='preco', y='quilometragem (Km)', color='marca', 
                             hover_data=['Ano', 'modelo', 'combustivel_Gasolina', 'Transmission_Manual'],
                             title='Gráfico de Dispersão: Preço x Quilometragem', 
                             color_discrete_map={brand: color for brand, color in zip(self.df['marca'].unique(), self.brand_colors)})

            # Atualizando o gráfico para exibir preço em R$
            fig.update_layout(
                xaxis_title="Preço (R$)",
                xaxis_tickprefix="R$ ",
                showlegend=False  # Removendo a legenda
            )

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
    app = CarAnalysisApp(data_path, exchange_rate=6.1651)  # Taxa de câmbio de 6,1651 reais por 1 dólar
    app.run_app()


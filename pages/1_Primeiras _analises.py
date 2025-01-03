import pandas as pd
import streamlit as st
import plotly.express as px

class CarAnalysisApp:
    def __init__(self, data_path, exchange_rate=6.1651):
        self.data_path = data_path
        self.df = None
        self.brand_colors = None
        self.selected_years = None
        self.selected_fuel = None
        self.exchange_rate = exchange_rate

    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            st.success("Dados carregados com sucesso!")
            self.convert_price_to_real()
            self.convert_years()
        except Exception as e:
            st.error(f"Erro ao carregar os dados: {e}")

    def convert_price_to_real(self):
        if 'preco' in self.df.columns:
            self.df['preco'] = self.df['preco'] * self.exchange_rate
        else:
            st.warning("Coluna 'preco' não encontrada!")

    def convert_years(self):
        if 'Ano' in self.df.columns:
            year_mapping = {i: 2000 + i - 1 for i in range(1, 25)}
            self.df['Ano'] = self.df['Ano'].map(year_mapping)
        else:
            st.warning("Coluna 'Ano' não encontrada!")

    def filter_top_10_brands(self):
        if self.df is not None:
            top_brands = self.df['marca'].value_counts().head(10).index
            self.df = self.df[self.df['marca'].isin(top_brands)]
            self.brand_colors = px.colors.qualitative.Set2[:10]
        else:
            st.warning("Nenhum dado carregado ainda!")

    def show_boxplot_by_quilometragem(self):
        st.subheader("Boxplot: Quilometragem por Marca")
        if self.df is not None:
            quilometragem_ticks = [100000, 200000, 300000, 400000, 500000]
            quilometragem_ticks_labels = ['100 Km', '200 Km', '300 Km', '400 Km', '500 Km']

            fig = px.box(self.df, x='marca', y='quilometragem (Km)', title='Boxplot das Marcas por Quilometragem', 
                         color='marca', color_discrete_map={brand: color for brand, color in zip(self.df['marca'].unique(), self.brand_colors)})

            fig.update_layout(
                yaxis_title="Quilometragem",
                yaxis=dict(
                    tickvals=quilometragem_ticks,
                    ticktext=quilometragem_ticks_labels
                ),
                showlegend=False
            )
            st.plotly_chart(fig)
        else:
            st.warning("Dados não disponíveis para exibição.")

    def show_histogram_by_brand(self):
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

            fig = px.bar(filtered_df, x='Ano', y='preco', color='marca', title='Relação entre Preço e Ano', 
                         color_discrete_map={brand: color for brand, color in zip(self.df['marca'].unique(), self.brand_colors)})

            fig.update_layout(
                yaxis_title="Preço (R$)", 
                yaxis_tickprefix="R$ ",
                showlegend=False
            )

            st.plotly_chart(fig)
        else:
            st.warning("Dados não disponíveis para exibição.")

    def show_scatter_plot(self):
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

            fig = px.scatter(filtered_df, x='preco', y='quilometragem (Km)', color='marca', 
                             hover_data=['Ano', 'modelo', 'combustivel_Gasolina', 'Transmission_Manual'],
                             title='Gráfico de Dispersão: Preço x Quilometragem', 
                             color_discrete_map={brand: color for brand, color in zip(self.df['marca'].unique(), self.brand_colors)})

            fig.update_layout(
                xaxis_title="Preço (R$)",
                xaxis_tickprefix="R$ ",
                showlegend=False
            )

            st.plotly_chart(fig)
        else:
            st.warning("Dados não disponíveis para exibição.")

    def run_app(self):
        st.title("Primeiras Análises")
        self.load_data()
        self.filter_top_10_brands()

        self.show_boxplot_by_quilometragem()
        self.show_histogram_by_brand()
        self.show_bar_chart_preco_ano()
        self.show_scatter_plot()

if __name__ == "__main__":
    data_path = "Datas/1_Cars_processado.csv"
    app = CarAnalysisApp(data_path, exchange_rate=6.1651)
    app.run_app()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

class CarAnalysisCharts:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        """Carrega os dados do CSV."""
        self.df = pd.read_csv(self.data_path)

    def plot_price_vs_kilometrage(self):
        """Gráfico de dispersão: Preço x Quilometragem."""
        fig = px.scatter(
            self.df, 
            x='quilometragem', 
            y='preco', 
            color='marca',
            size='preco',
            hover_data=['modelo', 'ano', 'combustivel', 'transmissão'],
            title='Relação entre Preço e Quilometragem',
        )
        fig.update_layout(xaxis_title='Quilometragem (Km)', yaxis_title='Preço (R$)')
        st.plotly_chart(fig)

    def plot_price_distribution_by_brand(self):
        """Gráfico de violino para distribuição de preços por marca."""
        fig = px.violin(
            self.df, 
            x='marca', 
            y='preco', 
            box=True, 
            points='all',
            color='marca',
            title='Distribuição de Preços por Marca',
        )
        fig.update_layout(xaxis_title='Marca', yaxis_title='Preço (R$)', showlegend=False)
        st.plotly_chart(fig)

    def plot_price_trends_over_years(self):
        """Gráfico de linhas para tendências de preços ao longo dos anos."""
        avg_price_per_year = self.df.groupby('ano')['preco'].mean().reset_index()
        fig = px.line(
            avg_price_per_year, 
            x='ano', 
            y='preco', 
            title='Tendências de Preços Médios ao Longo dos Anos',
            markers=True
        )
        fig.update_layout(xaxis_title='Ano', yaxis_title='Preço Médio (R$)')
        st.plotly_chart(fig)

    def plot_fuel_type_vs_price(self):
        """Gráfico de barras para preços médios por tipo de combustível."""
        avg_price_by_fuel = self.df.groupby('combustivel')['preco'].mean().reset_index()
        fig = px.bar(
            avg_price_by_fuel, 
            x='combustivel', 
            y='preco', 
            title='Preços Médios por Tipo de Combustível',
            color='combustivel',
        )
        fig.update_layout(xaxis_title='Tipo de Combustível', yaxis_title='Preço Médio (R$)', showlegend=False)
        st.plotly_chart(fig)

    def plot_price_by_transmission_type(self):
        """Gráfico de barras para preços médios por tipo de transmissão."""
        avg_price_by_trans = self.df.groupby('transmissão')['preco'].mean().reset_index()
        fig = px.bar(
            avg_price_by_trans, 
            x='transmissão', 
            y='preco', 
            title='Preços Médios por Tipo de Transmissão',
            color='transmissão',
        )
        fig.update_layout(xaxis_title='Tipo de Transmissão', yaxis_title='Preço Médio (R$)', showlegend=False)
        st.plotly_chart(fig)

    def run(self):
        """Executa os gráficos."""
        self.load_data()
        st.title('Análise Exploratória de Veículos Usados')
        st.sidebar.title('Opções de Análise')

        st.sidebar.subheader('Fatores que Influenciam no Preço')
        if st.sidebar.checkbox('Quilometragem x Preço'):
            self.plot_price_vs_kilometrage()
        if st.sidebar.checkbox('Distribuição de Preço por Marca'):
            self.plot_price_distribution_by_brand()

        st.sidebar.subheader('Tendências Históricas')
        if st.sidebar.checkbox('Tendências de Preço ao Longo dos Anos'):
            self.plot_price_trends_over_years()
        if st.sidebar.checkbox('Preço por Tipo de Combustível'):
            self.plot_fuel_type_vs_price()
        if st.sidebar.checkbox('Preço por Tipo de Transmissão'):
            self.plot_price_by_transmission_type()

# Caminho para o arquivo CSV
data_path = '/mnt/data/1_Cars_dataset_processado.csv'

# Executa a aplicação
if __name__ == '__main__':
    app = CarAnalysisCharts(data_path)
    app.run()

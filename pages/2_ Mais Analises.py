import pandas as pd
import streamlit as st
import plotly.express as px

# Função para converter valores de string para float
def convert_to_float(value):
    return float(str(value).replace('R$', '').replace('.', '').replace('.', ','))

# Função para limitar os valores de preço a 7 dígitos e formatar como Real Brasileiro
def format_to_brl_limited(value):
    """Formata o valor no padrão monetário brasileiro com ponto como separador de milhar e vírgula como separador decimal, limitado a 7 dígitos."""
    # Limitar o valor para no máximo 9.999.999
    if value > 9999999:
        value = 9999999
    # Formatar o valor para Real Brasileiro com ponto como separador de milhar e vírgula como separador decimal
    return f"R$ {value:,.2f}".replace(".", "X").replace(".", ",").replace("X", ",")

# Função para formatar os valores sem centavos no formato monetário brasileiro
def format_to_brl_without_cents(value):
    """Formata o valor no padrão monetário brasileiro (R$) sem centavos."""
    # Limitar o valor para no máximo 9.999.999
    if value > 9999999:
        value = 9999999
    # Arredonda para o valor inteiro e formata
    return f"R$ {value:,.0f}".replace(".", "X").replace(".", ",").replace("X", ",")

class CarAnalysisApp:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.df_filtered = None

    def load_data(self):
        """Carrega os dados do arquivo CSV."""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['preco'] = self.df['preco'].apply(convert_to_float)
            self.df_filtered = self.df.copy()
            st.sidebar.success("Dados carregados com sucesso!")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar os dados: {e}")

    def add_filters(self):
        """Adiciona filtros interativos no painel lateral."""
        if self.df is not None:
            anos = sorted(self.df['ano'].unique())
            ano_min, ano_max = st.sidebar.slider(
                "Selecione o intervalo de anos:", 
                min_value=int(min(anos)), 
                max_value=int(max(anos)), 
                value=(int(min(anos)), int(max(anos)))
            )
            preco_min, preco_max = st.sidebar.slider(
                "Selecione o intervalo de preços:", 
                min_value=int(self.df['preco'].min()), 
                max_value=int(self.df['preco'].max()), 
                value=(int(self.df['preco'].min()), int(self.df['preco'].max()))
            )
            self.df_filtered = self.df[(self.df['ano'] >= ano_min) & (self.df['ano'] <= ano_max) & 
                                       (self.df['preco'] >= preco_min) & (self.df['preco'] <= preco_max)]

    def show_histogram_year(self):
        """Histograma de distribuição de veículos por ano."""
        fig = px.histogram(
            self.df_filtered, x='ano', color='marca',
            title='DISTRIBUIÇÃO DE VEÍCULOS POR ANO',
            labels={'ano': 'ANO'},
        )
        fig.update_layout(
            yaxis_title='UNIDADES',
            showlegend=False
        )
        st.plotly_chart(fig)

    def show_boxplot_price_brand(self):
        """Boxplot de preços por marca."""
        # Atualizando os preços formatados
        self.df_filtered['preco_formatted'] = self.df_filtered['preco'].apply(lambda x: format_to_brl_limited(x))
        
        fig = px.box(
            self.df_filtered, x='marca', y='preco', color='marca',
            title='BOXPLOT DE PREÇOS POR MARCA',
            labels={'marca': 'MARCA', 'preco': 'PREÇO (R$)'}
        )
        
        fig.update_traces(
            hovertemplate=(
                "MARCA: %{x}<br>"
                "PREÇO (R$): %{y:,.2f}<extra></extra>"  # Exibindo com 2 casas decimais, mas no formato BR
            )
        )
        
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)

    def show_violin_price_transmission(self):
        """Gráfico de violino de preços por tipo de transmissão."""
        fig = px.violin(
            self.df_filtered, y='preco', x='tipo',
            title='PREÇOS POR TIPO DE VEÍCULO',
            labels={'tipo': 'TIPO', 'preco': 'PREÇO (R$)'}
        )
        
        fig.update_traces(
            hovertemplate=(
                "TIPO: %{x}<br>"
                "PREÇO (R$): %{y:,.2f}<extra></extra>"  # Exibindo com 2 casas decimais, mas no formato BR
            )
        )
        
        st.plotly_chart(fig)

    def show_bar_model_price(self):
        """Gráfico de barras de preço médio por modelo."""
        avg_price_by_model = self.df_filtered.groupby('modelo')['preco'].mean().reset_index().sort_values(by='preco', ascending=False)
        
        fig = px.bar(
            avg_price_by_model, x='modelo', y='preco',
            title='PREÇO MÉDIO POR MODELO',
            labels={'modelo': 'MODELO', 'preco': 'PREÇO MÉDIO (R$)'}
        )
        
        fig.update_traces(
            hovertemplate=( 
                "MODELO: %{x}<br>"
                "PREÇO MÉDIO (R$): %{y:,.2f}<extra></extra>"  # Exibindo com 2 casas decimais, mas no formato BR
            )
        )
        
        fig.update_layout(
            yaxis_tickvals=[1e6, 2e6, 3e6, 4e6, 5e6],
            yaxis_ticktext=["1.000.000", "2.000.000", "3.000.000", "4.000.000", "5.000.000"],
            xaxis_tickangle=-45,
            yaxis_title="PREÇO MÉDIO (R$)",
            title="Preço Médio por Modelo"
        )
        
        st.plotly_chart(fig)

    def show_density_price(self):
        """Gráfico de densidade do preço."""
        fig = px.density_contour(
            self.df_filtered, x='ano', y='preco',
            title='DENSIDADE DO PREÇO POR ANO',
            labels={'ano': 'ANO', 'preco': 'PREÇO (R$)'}
        )
        
        fig.update_traces(
            hovertemplate=( 
                "ANO: %{x:.0f}<br>"
                "PREÇO (R$): %{y:,.2f}<br>"  # Exibindo com 2 casas decimais, mas no formato BR
                "QUANT: %{z}<extra></extra>"
            )
        )
        
        st.plotly_chart(fig)

    def show_treemap_brand_model(self):
        """Mapa de árvore de distribuição de marcas e modelos pelo preço."""
        self.df_filtered['hover_info'] = (
            'MODELO: ' + self.df_filtered['modelo'] + '<br>' +
            'PREÇO (R$): ' + self.df_filtered['preco'].apply(lambda x: format_to_brl_limited(x)) + '<br>' +  # Usando a formatação limitada
            'MARCA: ' + self.df_filtered['marca']
        )
        
        fig = px.treemap(
            self.df_filtered, path=['marca', 'modelo'], values='preco',
            title='DISTRIBUIÇÃO DE MARCAS E MODELOS PELO PREÇO'
        )
        
        fig.update_traces(hovertemplate='%{customdata}<extra></extra>', customdata=self.df_filtered['hover_info'])
        
        st.plotly_chart(fig)

    def run_app(self):
        st.title("Análise Exploratória")
        self.load_data()
        self.add_filters()
        self.show_histogram_year()
        self.show_boxplot_price_brand()
        self.show_violin_price_transmission()
        self.show_bar_model_price()
        self.show_density_price()
        self.show_treemap_brand_model()

if __name__ == "__main__":
    data_path = "Datas/1_Cars_dataset_processado.csv"
    app = CarAnalysisApp(data_path)
    app.run_app()

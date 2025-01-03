import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Função para converter valores de string para float
def convert_to_float(value):
    return float(str(value).replace('R$', '').replace('.', '').replace(',', '.'))

class CarAnalysisApp:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.df_filtered = None  # DataFrame filtrado pelos anos

    def load_data(self):
        """Carrega os dados do arquivo CSV."""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['preco'] = self.df['preco'].apply(convert_to_float)
            self.df_filtered = self.df.copy()  # Inicializa o DataFrame filtrado
            st.sidebar.success("Dados carregados com sucesso!")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar os dados: {e}")

    def add_year_filter(self):
        """Adiciona um filtro de ano ao painel lateral e retorna o intervalo selecionado."""
        if self.df is not None:
            anos = sorted(self.df['ano'].unique())
            ano_min, ano_max = st.sidebar.slider(
                "Selecione o intervalo de anos:", 
                min_value=int(min(anos)), 
                max_value=int(max(anos)), 
                value=(int(min(anos)), int(max(anos)))
            )
            # Filtra o DataFrame com base nos anos selecionados
            self.df_filtered = self.df[(self.df['ano'] >= ano_min) & (self.df['ano'] <= ano_max)]
            return ano_min, ano_max
        return None, None

    def show_price_distribution_by_brand(self):
        """Distribuição de preços por marca com gráfico de dispersão."""
        if self.df_filtered is not None:
            fig = px.scatter(
                self.df_filtered, x='marca', y='preco', 
                color='marca',
                title='Distribuição de Preços por Marca (Dispersão)',
                labels={'marca': 'Marca', 'preco': 'Preço (R$)'}
            )
            # Removendo a legenda "Marca" do lado direito do gráfico
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

    def show_price_trends_over_years(self):
        """Tendências de preços médios ao longo dos anos."""
        if self.df_filtered is not None:
            avg_price_by_year = self.df_filtered.groupby('ano')['preco'].mean().reset_index()
            fig = px.line(
                avg_price_by_year, x='ano', y='preco', 
                title='Tendência de Preços Médios ao Longo dos Anos',
                labels={'ano': 'Ano', 'preco': 'Preço Médio (R$)'}
            )
            st.plotly_chart(fig)

    def show_price_by_fuel_type(self):
        """Preços médios por tipo de combustível com gráfico de barras."""
        if self.df_filtered is not None:
            avg_price_by_fuel = self.df_filtered.groupby('combustivel')['preco'].mean().reset_index()
            fig = px.bar(
                avg_price_by_fuel, x='combustivel', y='preco', 
                color='combustivel',
                title='Preços Médios por Tipo de Combustível',
                labels={'combustivel': 'Combustível', 'preco': 'Preço Médio (R$)'}
            )
            st.plotly_chart(fig)

    def show_price_by_transmission_type(self):
        """Preços médios por tipo de transmissão com gráfico de barras."""
        if self.df_filtered is not None:
            # Substituindo 'Importado' e 'Nacional' por 'Manual' e 'Automático'
            self.df_filtered['tipo'] = self.df_filtered['tipo'].replace({'Importado': 'Manual', 'Nacional': 'Automático'})
            
            avg_price_by_transmission = self.df_filtered.groupby('tipo')['preco'].mean().reset_index()
            fig = px.bar(
                avg_price_by_transmission, x='tipo', y='preco', 
                color='tipo',
                title='Preços Médios por Tipo de Transmissão',
                labels={'tipo': 'Tipo de Transmissão', 'preco': 'Preço Médio (R$)'}
            )
            st.plotly_chart(fig)

    def show_price_by_fuel_and_brand(self):
        """Gráfico de barras empilhadas para preços por tipo de combustível e marca."""
        if self.df_filtered is not None:
            avg_price_by_fuel_brand = self.df_filtered.groupby(['combustivel', 'marca'])['preco'].mean().reset_index()
            fig = px.bar(
                avg_price_by_fuel_brand, 
                x='combustivel', 
                y='preco', 
                color='marca', 
                title='Preços por Tipo de Combustível e Marca (Barras Empilhadas)', 
                labels={'combustivel': 'Tipo de Combustível', 'preco': 'Preço Médio (R$)'},
                barmode='stack'  # Empilhando as barras
            )
            # Retirando a legenda 'Marca' do lado direito
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

    def run_app(self):
        """Executa todos os métodos da aplicação."""
        st.title("Análise Exploratória de Carros Usados")
        
        # Título adicional na sidebar
        st.sidebar.title("Opções de Análise")
        st.sidebar.subheader("Fatores que Influenciam no Preço")

        self.load_data()

        # Mostrar filtro de anos
        ano_min, ano_max = self.add_year_filter()

        # Adicionando botão de "Atualizar Gráficos"
        if st.sidebar.button('Atualizar Gráficos'):
            # Somente atualizar gráficos quando o botão for pressionado
            if ano_min and ano_max:
                self.show_price_distribution_by_brand()
                self.show_price_trends_over_years()
                self.show_price_by_fuel_type()
                self.show_price_by_transmission_type()
                self.show_price_by_fuel_and_brand()
            else:
                st.sidebar.warning("Selecione um intervalo de anos para atualizar os gráficos.")

# Caminho do arquivo CSV
data_path = "Datas/1_Cars_dataset_processado.csv"

# Inicializa o aplicativo
if __name__ == "__main__":
    app = CarAnalysisApp(data_path)
    app.run_app()

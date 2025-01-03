import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Classe principal para análise exploratória de dados (EDA)
class CarEDAApp:
    def __init__(self, data):
        self.df = data
        self.create_sidebar()
        self.create_main_panel()

    def create_sidebar(self):
        st.sidebar.header('Configurações do Gráfico')
        self.plot_type = st.sidebar.selectbox(
            'Escolha o tipo de gráfico:',
            ['Histograma', 'Boxplot', 'Scatterplot', 'Countplot']
        )
        self.column = st.sidebar.selectbox('Escolha a coluna para análise:', self.df.columns)

    def create_main_panel(self):
        st.title('Análise Exploratória de Dados de Carros')
        self.generate_all_plots()

    def generate_all_plots(self):
        st.subheader('Distribuição das Variáveis')
        plt.figure(figsize=(10, 6))
        for column in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                plt.figure(figsize=(10, 6))
                sns.histplot(self.df[column], kde=True)
                plt.title(f'Histograma de {column}')
                st.pyplot(plt)

        st.subheader('Boxplots das Variáveis Numéricas')
        for column in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=self.df[column])
                plt.title(f'Boxplot de {column}')
                st.pyplot(plt)
        
        st.subheader('Distribuição de Categorias')
        for column in self.df.columns:
            if pd.api.types.is_categorical_dtype(self.df[column]) or self.df[column].dtype == 'object':
                plt.figure(figsize=(10, 6))
                sns.countplot(x=self.df[column])
                plt.title(f'Countplot de {column}')
                st.pyplot(plt)

if __name__ == '__main__':
    st.title('Upload de Dataset de Carros')
    uploaded_file = st.file_uploader('Carregue o arquivo CSV com os dados dos carros', type='csv')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        app = CarEDAApp(df)  # Inicializa o aplicativo
        st.write('Aplicativo de análise de dados iniciado.')

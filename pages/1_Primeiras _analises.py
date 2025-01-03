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
        self.display_data()
        self.generate_plot()

    def display_data(self):
        st.subheader('Visualização Inicial dos Dados')
        st.write(self.df.head())
        st.write(self.df.describe())

    def generate_plot(self):
        st.subheader('Gráfico Gerado')
        plt.figure(figsize=(10, 6))
        
        if self.plot_type == 'Histograma':
            sns.histplot(self.df[self.column], kde=True)
        elif self.plot_type == 'Boxplot':
            sns.boxplot(x=self.df[self.column])
        elif self.plot_type == 'Scatterplot':
            x_col = st.sidebar.selectbox('Escolha a coluna X:', self.df.columns)
            y_col = st.sidebar.selectbox('Escolha a coluna Y:', self.df.columns)
            sns.scatterplot(x=self.df[x_col], y=self.df[y_col])
        elif self.plot_type == 'Countplot':
            sns.countplot(x=self.df[self.column])
        
        st.pyplot(plt)

if __name__ == '__main__':
    file_path = 'Datas/1_Cars_processado.csv'
    df = pd.read_csv(file_path)
    app = CarEDAApp(df)  # Inicializa o aplicativo
    st.write('Aplicativo de análise de dados iniciado.')

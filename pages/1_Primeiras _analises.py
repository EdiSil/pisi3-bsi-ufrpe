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
        self.plot_type = st.sidebar.multiselect(
            'Escolha os tipos de gráficos:',
            ['Histograma', 'Boxplot', 'Countplot'],
            default=['Histograma', 'Boxplot', 'Countplot']
        )

    def create_main_panel(self):
        st.title('Análise Exploratória de Dados de Carros')
        self.generate_all_plots()

    def generate_all_plots(self):
        if 'Histograma' in self.plot_type:
            st.subheader('Distribuição das Variáveis Numéricas (Histogramas)')
            for column in self.df.select_dtypes(include=['float64', 'int64']).columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(self.df[column], kde=True)
                plt.title(f'Histograma de {column}')
                st.pyplot(plt)
        
        if 'Boxplot' in self.plot_type:
            st.subheader('Boxplots das Variáveis Numéricas')
            for column in self.df.select_dtypes(include=['float64', 'int64']).columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=self.df[column])
                plt.title(f'Boxplot de {column}')
                st.pyplot(plt)
        
        if 'Countplot' in self.plot_type:
            st.subheader('Distribuição de Categorias (Countplots)')
            for column in self.df.select_dtypes(include=['object', 'category']).columns:
                plt.figure(figsize=(10, 6))
                sns.countplot(x=self.df[column])
                plt.title(f'Countplot de {column}')
                plt.xticks(rotation=45)
                st.pyplot(plt)

if __name__ == '__main__':
    st.title('Análise Exploratória de Dados de Carros')
    file_path = 'Datas/1_Cars_processado.csv'
    df = pd.read_csv(file_path)
    app = CarEDAApp(df)  # Inicializa o aplicativo
    st.write('Aplicativo de análise de dados iniciado.')

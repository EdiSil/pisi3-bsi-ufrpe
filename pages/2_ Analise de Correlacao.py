import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

# Função para carregar os dados a partir de uma URL
def load_data():
    url = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_dataset002.csv"
    df = pd.read_csv(url)
    return df

# Função para calcular e exibir a matriz de correlação
def plot_correlation_matrix(df):
    # Selecionando as colunas relevantes
    df_corr = df[['marca', 'modelo', 'ano', 'quilometragem', 'preco', 'combustivel', 'tipo']].copy()
    
    # Convertendo para tipo numérico
    df_corr['ano'] = pd.to_numeric(df_corr['ano'], errors='coerce')
    df_corr['quilometragem'] = pd.to_numeric(df_corr['quilometragem'], errors='coerce')
    df_corr['preco'] = pd.to_numeric(df_corr['preco'], errors='coerce')
    
    # Calculando a correlação
    correlation_matrix = df_corr.corr()

    # Criando o heatmap interativo com Plotly
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis',  # Paleta de cores mais profissional
        colorbar=dict(title="Correlação"),
        hoverongaps=False
    ))

    fig.update_layout(
        title="Matriz de Correlação",
        xaxis_title="Variáveis",
        yaxis_title="Variáveis",
        width=800,
        height=800,
        template="plotly_dark"  # Usando um template escuro para um visual mais elegante
    )

    st.plotly_chart(fig)

# Função principal para executar o aplicativo Streamlit
def app():
    st.title('Análise de Dados de Carros - Matriz de Correlação')

    # Carregar os dados
    df = load_data()

    # Exibir uma amostra dos dados
    st.write("Pré-visualização dos Dados Carregados:")
    st.dataframe(df.head())

    # Exibir a Matriz de Correlação
    st.header('Matriz de Correlação')
    plot_correlation_matrix(df)

# Executando a aplicação Streamlit
if __name__ == "__main__":
    app()

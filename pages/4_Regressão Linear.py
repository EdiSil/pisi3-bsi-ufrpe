import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Função para carregar o arquivo CSV
def carregar_arquivo():
    """
    Função para carregar o dataset diretamente de uma URL ou caminho local.
    """
    url_csv = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_novo.csv"
    try:
        data = pd.read_csv(url_csv)
        return data
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

# Função para treinar o modelo de Regressão Linear e avaliar o desempenho
def treinar_modelo_e_avaliar(data, variaveis, alvo):
    """
    Função para treinar o modelo de Regressão Linear e avaliar o desempenho.
    
    Parâmetros:
    data (DataFrame): DataFrame com as variáveis de entrada e a variável alvo.
    variaveis (list): Lista de colunas para as variáveis preditoras.
    alvo (str): Nome da coluna que representa a variável alvo.
    
    Retorna:
    model: O modelo LinearRegression treinado.
    X_train, X_test, y_train, y_test: Conjuntos de dados para treinamento e teste.
    y_pred: Previsões feitas pelo modelo.
    rmse (float): Erro quadrático médio (RMSE).
    r2 (float): Coeficiente de determinação (R²).
    """
    # Preparar os dados
    X = data[variaveis]  # Variáveis preditoras
    y = data[alvo]       # Variável alvo (preço dos carros)

    # Dividir os dados em conjunto de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Criar e treinar o modelo de Regressão Linear
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Avaliar o desempenho do modelo usando RMSE e R²
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    return model, X_train, X_test, y_train, y_test, y_pred, rmse, r2

# Função para exibir o gráfico de comparação entre valores reais e previstos com opções de sobreposição
def grafico_comparacao(y_test, y_pred, adicionar_linha_ideal, adicionar_linha_tendencia):
    """
    Função para exibir o gráfico de dispersão (scatter plot) dos valores reais e previstos com opções de sobreposição.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotando o gráfico de dispersão
    ax.scatter(y_test, y_pred, color='blue', alpha=0.5, label="Valores Reais vs. Previstos")
    
    # Adicionando a linha ideal (y = x), se o usuário escolher
    if adicionar_linha_ideal:
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label="Linha Ideal (y = x)")
    
    # Adicionando uma linha de tendência (regressão linear) se o usuário escolher
    if adicionar_linha_tendencia:
        sns.regplot(x=y_test, y=y_pred, scatter=False, ax=ax, color='green', line_kws={"lw": 2, "ls": "--"}, label="Linha de Tendência")
    
    ax.set_title("Comparação entre Valores Reais e Previstos")
    ax.set_xlabel("Valor Real")
    ax.set_ylabel("Valor Previsto")
    ax.legend()
    st.pyplot(fig)

# Função para exibir a tabela de comparação entre valores reais e previstos
def tabela_comparacao(y_test, y_pred):
    """
    Função para exibir uma tabela formatada de comparação entre os valores reais e previstos.
    """
    comparacao = pd.DataFrame({'Valor Real': y_test, 'Valor Previsto': y_pred})
    comparacao['Erro Absoluto'] = abs(comparacao['Valor Real'] - comparacao['Valor Previsto'])
    comparacao['Erro Relativo'] = comparacao['Erro Absoluto'] / comparacao['Valor Real'] * 100
    st.write("### Tabela de Comparação (Valores Reais vs. Previstos):")
    st.write(comparacao.head(20))  # Mostrar os 20 primeiros resultados

# Função principal da aplicação Streamlit
def main():
    st.title("Previsão de Preços de Carros com Regressão Linear")

    # Carregar o dataset
    data = carregar_arquivo()

    if data is not None:
        # Exibir as primeiras linhas dos dados
        st.write("### Dados Carregados:")
        st.write(data.head())

        # Definir as variáveis preditoras e alvo
        variaveis = ['Year', "KM's driven", 'Fuel_Diesel', 'Fuel_Petrol', 'Assembly_Local', 'Transmission_Manual']
        alvo = 'Price'

        # Verificar se todas as colunas necessárias estão presentes
        for col in variaveis + [alvo]:
            if col not in data.columns:
                st.error(f"A coluna '{col}' não foi encontrada no dataset.")
                return

        # Treinar o modelo e avaliar desempenho
        model, X_train, X_test, y_train, y_test, y_pred, rmse, r2 = treinar_modelo_e_avaliar(
            data, variaveis, alvo
        )

        # Exibir as métricas de avaliação
        st.write(f"### Resultados da Avaliação do Modelo:")
        st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        # Comparar valores reais e previstos (top 20)
        tabela_comparacao(y_test, y_pred)

        # Opções de sobreposição no gráfico
        adicionar_linha_ideal = st.checkbox("Adicionar Linha Ideal (y = x)", value=True)
        adicionar_linha_tendencia = st.checkbox("Adicionar Linha de Tendência (Regressão Linear)", value=False)

        # Exibir o gráfico de comparação Real vs Previsto com as opções escolhidas
        grafico_comparacao(y_test, y_pred, adicionar_linha_ideal, adicionar_linha_tendencia)

# Executando a aplicação Streamlit
if __name__ == "__main__":
    main()

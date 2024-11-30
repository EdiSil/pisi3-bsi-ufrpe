import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Função para treinar o modelo Random Forest e avaliar o desempenho
def treinar_modelo_e_avaliar(descricao_modelo, data, variaveis, alvo):
    """
    Função para treinar o modelo Random Forest e avaliar seu desempenho.
    
    Parâmetros:
    descricao_modelo (str): Nome ou descrição do modelo.
    data (DataFrame): DataFrame com as variáveis de entrada e a variável alvo.
    variaveis (list): Lista de colunas para as variáveis preditoras.
    alvo (str): Nome da coluna que representa a variável alvo.
    
    Retorna:
    model: O modelo RandomForestRegressor treinado.
    X_train, X_test, y_train, y_test: Conjuntos de dados para treinamento e teste.
    y_pred: Previsões feitas pelo modelo.
    rmse (float): Erro quadrático médio (RMSE).
    r2 (float): Coeficiente de determinação (R²).
    """
    
    # 1. Preparar os dados de entrada e saída
    X = data[variaveis]  # Variáveis preditoras
    y = data[alvo]       # Variável alvo (preço dos carros)

    # 2. Dividir os dados em conjunto de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Criar e treinar o modelo Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Fazer previsões com o conjunto de teste
    y_pred = model.predict(X_test)
    
    # 5. Avaliar o desempenho do modelo usando métricas
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE (Root Mean Squared Error)
    r2 = r2_score(y_test, y_pred)  # R² (Coeficiente de Determinação)
    
    # 6. Exibir as métricas de desempenho
    st.write(f"\nDesempenho do Modelo: {descricao_modelo}")
    st.write(f"RMSE (Erro Quadrático Médio): {rmse:.2f}")
    st.write(f"R² (Coeficiente de Determinação): {r2:.2f}")
    
    # 7. Exibir gráfico de comparação entre valores reais e previstos
    comparar_df = pd.DataFrame({'Real': y_test, 'Previsto': y_pred})
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=comparar_df, ax=ax)
    ax.set_title(f"Comparação entre Valores Reais e Previstos - {descricao_modelo}")
    ax.set_xlabel("Índice")
    ax.set_ylabel("Preço")
    ax.legend(title="Legenda", labels=["Real", "Previsto"])
    st.pyplot(fig)
    
    # 8. Exibir gráfico de importância das variáveis no modelo
    importancia_features = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importancia_features, ax=ax)
    ax.set_title(f'Importância das Features - {descricao_modelo}')
    ax.set_xlabel('Importância')
    ax.set_ylabel('Características')
    st.pyplot(fig)

    # Retornar o modelo treinado e outras variáveis úteis
    return model, X_train, X_test, y_train, y_test, y_pred, rmse, r2

# Função para carregar o arquivo CSV diretamente
def carregar_arquivo():
    # Carregar o arquivo diretamente de uma URL ou caminho local
    file_path = 'OLX_cars_novo.csv'  # Caminho para o arquivo CSV
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

# Função principal da aplicação Streamlit
def main():
    st.title("Classificação de Preços de Carros com Random Forest")

    # Carregar o dataset
    data = carregar_arquivo()

    if data is not None:
        # Mostrar as primeiras linhas dos dados
        st.write("### Dados Carregados:")
        st.write(data.head())

        # Pré-processar os dados (defina suas variáveis de entrada e alvo)
        variaveis = ['Year', "KM's driven", 'Fuel', 'Assembly', 'Transmission']
        alvo = 'Price'

        # Treinar o modelo e avaliar desempenho
        descricao_modelo = "Modelo Random Forest para Preço de Carros"
        model, X_train, X_test, y_train, y_test, y_pred, rmse, r2 = treinar_modelo_e_avaliar(
            descricao_modelo, data, variaveis, alvo
        )

        # Exibir as métricas de avaliação
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"R²: {r2:.2f}")

# Executando a aplicação Streamlit
if __name__ == "__main__":
    main()

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Função para preparar os dados
def preparar_dados(file_path):
    # Carregar os dados do CSV
    data = pd.read_csv(file_path)

    # Selecionar colunas relevantes
    relevant_columns = ['Year', "KM's driven", 'Price', 'Fuel', 'Assembly', 'Transmission']
    filtered_data = data[relevant_columns].copy()

    # Tratar valores ausentes e duplicados
    filtered_data = filtered_data.drop_duplicates()

    # Tratar outliers (ano maior que 2024 ou km e preço acima do 99º percentil)
    filtered_data = filtered_data[filtered_data['Year'] <= 2024]  # Remover anos futuros
    km_99 = filtered_data["KM's driven"].quantile(0.99)
    price_99 = filtered_data["Price"].quantile(0.99)
    filtered_data = filtered_data[
        (filtered_data["KM's driven"] <= km_99) & 
        (filtered_data["Price"] <= price_99)
    ]

    # Converter variáveis categóricas para numéricas
    encoded_data = pd.get_dummies(filtered_data, columns=['Fuel', 'Assembly', 'Transmission'], drop_first=True)

    return encoded_data

# Função para treinar o modelo e avaliar o desempenho
def treinar_modelo_e_avaliar(X_train, X_test, y_train, y_test):
    # Criar o modelo Random Forest
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # Fazer previsões nos dados de teste
    y_pred = rf_model.predict(X_test)

    # Avaliar a performance do modelo
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # Raiz do Erro Quadrático Médio
    r2 = r2_score(y_test, y_pred)  # Coeficiente R²

    return rf_model, y_pred, rmse, r2

# Função principal da aplicação Streamlit
def main():
    st.title("Modelo de Preço de Carros com Random Forest")

    # Carregar o dataset
    file_path = 'OLX_cars_novo.csv'  # Caminho para o arquivo CSV
    st.write(f"### Carregando o arquivo: {file_path}")
    
    # Preparar os dados
    encoded_data = preparar_dados(file_path)

    # Mostrar as primeiras linhas dos dados preparados
    st.write("### Dados Carregados e Preparados:")
    st.write(encoded_data.head())

    # Dividir entre variáveis preditoras (X) e alvo (y)
    X = encoded_data.drop(columns=['Price'])
    y = encoded_data['Price']

    # Dividir os dados em treino (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo e avaliar o desempenho
    rf_model, y_pred, rmse, r2 = treinar_modelo_e_avaliar(X_train, X_test, y_train, y_test)

    # Exibir as métricas de desempenho
    st.write(f"### Avaliação do Modelo Random Forest:")
    st.write(f"**RMSE (Raiz do Erro Quadrático Médio):** {rmse:.2f}")
    st.write(f"**R² (Coeficiente de Determinação):** {r2:.2f}")

    # Exibir gráfico de comparação entre os valores reais e previstos
    st.write("### Comparação entre Valores Reais e Previstos:")
    comparar_df = pd.DataFrame({'Real': y_test, 'Previsto': y_pred})
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=comparar_df, ax=ax)
    ax.set_title("Comparação entre Valores Reais e Previstos")
    ax.set_xlabel("Índice")
    ax.set_ylabel("Preço")
    ax.legend(title="Legenda", labels=["Real", "Previsto"])
    st.pyplot(fig)

    # Exibir gráfico de importância das features
    st.write("### Importância das Features no Modelo:")
    importancia_features = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importancia_features, ax=ax)
    ax.set_title('Importância das Variáveis no Modelo')
    ax.set_xlabel('Importância')
    ax.set_ylabel('Características')
    st.pyplot(fig)

# Executando a aplicação Streamlit
if __name__ == "__main__":
    main()

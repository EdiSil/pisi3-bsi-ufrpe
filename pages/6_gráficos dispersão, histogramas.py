import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Função para carregar os dados diretamente do GitHub
def carregar_dados():
    url_csv = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_novo.csv"
    try:
        data = pd.read_csv(url_csv)
        return data
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None

# Função para preparar os dados
def preparar_dados(data):
    """
    Prepara o dataset para análise e modelagem:
    - Filtra colunas relevantes
    - Remove duplicatas
    - Trata valores ausentes e outliers
    - Codifica variáveis categóricas
    """
    relevant_columns = ['Year', "KM's driven", 'Price', 'Fuel_Diesel', 'Fuel_Petrol', 'Assembly_Local', 'Transmission_Manual']
    filtered_data = data[relevant_columns].copy()

    # Remover duplicados
    filtered_data = filtered_data.drop_duplicates()

    # Tratar outliers
    filtered_data = filtered_data[filtered_data['Year'] <= 2024]  # Remover anos futuros
    km_99 = filtered_data["KM's driven"].quantile(0.99)
    price_99 = filtered_data["Price"].quantile(0.99)
    filtered_data = filtered_data[
        (filtered_data["KM's driven"] <= km_99) & 
        (filtered_data["Price"] <= price_99)
    ]

    # Codificar variáveis categóricas
    encoded_data = pd.get_dummies(filtered_data, columns=['Fuel_Diesel', 'Fuel_Petrol', 'Assembly_Local', 'Transmission_Manual'], drop_first=True)

    return encoded_data

# Função para treinar o modelo e avaliar o desempenho
def treinar_modelo_e_avaliar(X_train, X_test, y_train, y_test):
    """
    Treina um modelo Random Forest e avalia o desempenho nos dados de teste.
    Retorna o modelo treinado, previsões, RMSE, R² e o conjunto de teste original.
    """
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # Fazer previsões nos dados de teste
    y_pred = rf_model.predict(X_test)

    # Avaliar a performance do modelo
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # Raiz do Erro Quadrático Médio
    r2 = r2_score(y_test, y_pred)  # Coeficiente R²

    return rf_model, y_pred, rmse, r2, y_test

# Função principal para a aplicação Streamlit
def main():
    st.title("Modelo de Preço de Carros com Random Forest")
    st.write("Esta aplicação utiliza um modelo de Random Forest para prever preços de carros com base em características fornecidas.")

    # Carregar os dados
    data = carregar_dados()

    if data is not None:
        # Mostrar as primeiras linhas do dataset original
        st.write("### Dataset Original:")
        st.write(data.head())

        # Preparar os dados
        encoded_data = preparar_dados(data)

        # Mostrar as primeiras linhas dos dados preparados
        st.write("### Dados Preparados:")
        st.write(encoded_data.head())

        # Dividir entre variáveis preditoras (X) e alvo (y)
        X = encoded_data.drop(columns=['Price'])
        y = encoded_data['Price']

        # Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Treinar o modelo e avaliar o desempenho
        rf_model, y_pred, rmse, r2, y_test = treinar_modelo_e_avaliar(X_train, X_test, y_train, y_test)

        # Exibir as métricas de desempenho
        st.write("### Avaliação do Modelo:")
        st.write(f"**RMSE (Raiz do Erro Quadrático Médio):** {rmse:.2f}")
        st.write(f"**R² (Coeficiente de Determinação):** {r2:.2f}")

        # Comparar valores reais e previstos
        comparison = pd.DataFrame({'Valor Real': y_test.values, 'Valor Previsto': y_pred})
        st.write("### Comparação entre Valores Reais e Previstos (Top 10):")
        st.write(comparison.head(10))

        # Visualização 1: Gráfico de Dispersão
        st.write("### Gráfico de Dispersão (Real vs. Previsto)")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, ax=ax1)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
        ax1.set_title('Valores Reais vs. Valores Previstos')
        ax1.set_xlabel('Valor Real')
        ax1.set_ylabel('Valor Previsto')
        st.pyplot(fig1)

        # Visualização 2: Histograma dos Erros
        st.write("### Distribuição dos Erros (Resíduos)")
        residuals = y_test - y_pred
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.histplot(residuals, bins=30, kde=True, color='blue', ax=ax2)
        ax2.set_title('Distribuição dos Erros')
        ax2.set_xlabel('Erro (Valor Real - Valor Previsto)')
        ax2.set_ylabel('Frequência')
        st.pyplot(fig2)
    else:
        st.error("Não foi possível carregar os dados. Verifique o endereço ou o formato do dataset.")

# Executar a aplicação Streamlit
if __name__ == "__main__":
    main()

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Função para carregar o arquivo CSV diretamente da URL
def carregar_arquivo():
    url_csv = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_novo.csv"
    try:
        data = pd.read_csv(url_csv)
        return data
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

# Função para treinar o modelo Random Forest e avaliar o desempenho
def treinar_modelo_e_avaliar(data, variaveis, alvo):
    X = data[variaveis]
    y = data[alvo]

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Fazer previsões
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    return model, X_train, X_test, y_train, y_test, y_pred, rmse, r2

# Função para salvar o modelo
def salvar_modelo(model, nome_arquivo="modelo_random_forest.pkl"):
    with open(nome_arquivo, 'wb') as arquivo:
        pickle.dump(model, arquivo)
    st.success(f"Modelo salvo como {nome_arquivo}")

# Função para carregar o modelo salvo
def carregar_modelo(nome_arquivo="modelo_random_forest.pkl"):
    try:
        with open(nome_arquivo, 'rb') as arquivo:
            model = pickle.load(arquivo)
        return model
    except FileNotFoundError:
        st.warning("Modelo salvo não encontrado. Treine e salve o modelo antes de carregá-lo.")
        return None

# Função para exibir o gráfico de comparação entre valores reais e previstos
def grafico_comparacao(y_test, y_pred):
    comparar_df = pd.DataFrame({'Real': y_test, 'Previsto': y_pred})
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=comparar_df, ax=ax)
    ax.set_title("Comparação entre Valores Reais e Previstos")
    ax.set_xlabel("Índice")
    ax.set_ylabel("Preço")
    ax.legend(title="Legenda", labels=["Real", "Previsto"])
    st.pyplot(fig)

# Função para exibir o gráfico de importância das variáveis
def grafico_importancia_features(X, model):
    importancia_features = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importancia_features, ax=ax)
    ax.set_title('Importância das Features')
    ax.set_xlabel('Importância')
    ax.set_ylabel('Características')
    st.pyplot(fig)

# Função principal da aplicação Streamlit
def main():
    st.title("Modelo Random Forest para Previsão de Preços de Carros")

    # Carregar os dados
    data = carregar_arquivo()

    if data is not None:
        st.write("### Dados Carregados:")
        st.write(data.head())

        # Definir variáveis preditoras e alvo
        variaveis = ['Year', "KM's driven", 'Fuel_Diesel', 'Fuel_Petrol', 'Assembly_Local', 'Transmission_Manual']
        alvo = 'Price'

        # Treinar o modelo
        st.write("### Treinando o Modelo...")
        model, X_train, X_test, y_train, y_test, y_pred, rmse, r2 = treinar_modelo_e_avaliar(data, variaveis, alvo)
        
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**R²:** {r2:.2f}")

        # Salvar o modelo treinado
        salvar_modelo(model)

        # Visualização dos dados
        opcao_grafico = st.selectbox("Escolha uma visualização:", 
                                     ["Comparação Real vs Previsto", "Importância das Features"])

        if opcao_grafico == "Comparação Real vs Previsto":
            grafico_comparacao(y_test, y_pred)
        elif opcao_grafico == "Importância das Features":
            grafico_importancia_features(X_train, model)

        # Predição com novos dados
        st.write("### Prever Preço com Novos Dados")
        km = st.number_input("KM driven", min_value=0.0, step=1000.0)
        ano = st.number_input("Ano do Carro", min_value=1900, max_value=2024, step=1)

        if st.button("Prever Preço"):
            modelo_carregado = carregar_modelo()
            if modelo_carregado:
                nova_predicao = modelo_carregado.predict([[ano, km]])
                st.write(f"**Preço Previsto:** R$ {nova_predicao[0]:,.2f}")

# Executar a aplicação Streamlit
if __name__ == "__main__":
    main()

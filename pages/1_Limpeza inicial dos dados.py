
import pandas as pd
import streamlit as st

# Função para carregar o arquivo CSV
def carregar_arquivo():
    # Carregar o arquivo CSV usando o uploader do Streamlit
    arquivo = st.file_uploader("https://github.com/EdiSil/pisi3-bsi-ufrpe/blob/main/data/OLX_cars_dataset00.csv", type=["csv"])
    if arquivo is not None:
        # Lê o arquivo CSV e retorna um DataFrame
        return pd.read_csv(arquivo)
    return None

# Função para realizar o pré-processamento e limpeza dos dados
def limpar_dados(data):
    # Selecionar colunas relevantes
    relevant_columns = [
        'Year', "KM's driven", 'Price', 'Fuel', 'Assembly', 'Transmission', 'Condition'
    ]
    filtered_data = data[relevant_columns].copy()

    # Remover duplicados
    filtered_data = filtered_data.drop_duplicates()

    # Filtrar anos no futuro (considerar máximo o ano atual)
    filtered_data = filtered_data[filtered_data['Year'] <= 2024]

    # Limitar quilometragem e preço com base no percentil 99 para remover outliers extremos
    km_99 = filtered_data["KM's driven"].quantile(0.99)
    price_99 = filtered_data["Price"].quantile(0.99)
    filtered_data = filtered_data[
        (filtered_data["KM's driven"] <= km_99) &
        (filtered_data["Price"] <= price_99)
    ]

    # Codificar variáveis categóricas
    encoded_data = pd.get_dummies(filtered_data, columns=['Fuel', 'Assembly', 'Transmission'], drop_first=True)

    return encoded_data

# Função para salvar o DataFrame limpo
def salvar_arquivo(data):
    # Salvar o DataFrame como um arquivo CSV
    data.to_csv('OLX_cars_novo.csv', index=False)
    st.success("Arquivo limpo salvo com sucesso como 'OLX_cars_novo.csv'.")

# Função principal
def main():
    st.title("Processamento de Dados de Carros Usados")

    # Carregar arquivo
    data = carregar_arquivo()

    if data is not None:
        st.write("Dados originais:", data.head())

        # Limpar dados
        data_limpa = limpar_dados(data)

        # Mostrar os dados limpos
        st.write("Dados limpos:", data_limpa.head())

        # Salvar os dados limpos
        salvar_arquivo(data_limpa)

# Executar o Streamlit
if __name__ == "__main__":
    main()

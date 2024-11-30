import pandas as pd
import streamlit as st

# URL do arquivo CSV no GitHub
URL_CSV = "https://raw.githubusercontent.com/EdiSil/pisi3-bsi-ufrpe/main/data/OLX_cars_dataset00.csv"

# Função para carregar o arquivo CSV
def carregar_arquivo():
    try:
        # Carregar o arquivo CSV diretamente da URL
        data = pd.read_csv(URL_CSV)
        return data
    except Exception as e:
        st.error("Erro ao carregar o arquivo. Verifique o link ou o formato do arquivo.")
        st.write(e)
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
    st.download_button(
        label="Baixar arquivo limpo",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name='OLX_cars_novo.csv',
        mime='text/csv'
    )

# Função principal
def main():
    st.title("Processamento de Dados de Carros Usados")

    # Carregar arquivo
    data = carregar_arquivo()

    if data is not None:
        st.write("### Dados originais:", data.head())

        # Limpar dados
        data_limpa = limpar_dados(data)

        # Mostrar os dados limpos
        st.write("### Dados limpos:", data_limpa.head())

        # Salvar os dados limpos
        salvar_arquivo(data_limpa)

# Executar o Streamlit
if __name__ == "__main__":
    main()

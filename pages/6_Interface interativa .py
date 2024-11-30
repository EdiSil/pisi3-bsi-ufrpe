import streamlit as st
import pickle
import numpy as np

# Função para carregar o modelo treinado
@st.cache_resource
def carregar_modelo(caminho_modelo):
    """
    Carrega o modelo treinado de um arquivo pickle.

    Args:
        caminho_modelo (str): Caminho para o arquivo do modelo.

    Returns:
        object: Modelo treinado.
    """
    try:
        with open(caminho_modelo, 'rb') as file:
            modelo = pickle.load(file)
        return modelo
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Função principal para a aplicação
def main():
    st.title("Previsão de Preço de Carros 🚗")
    st.markdown(
        """
        Esta aplicação utiliza um modelo de Machine Learning para prever o preço de carros com base em:
        - **KM driven (quilômetros rodados)**  
        - **Ano do carro**  
        
        **Insira os dados abaixo para obter uma estimativa.**
        """
    )

    # Entrada de dados do usuário
    km = st.number_input(
        "Quilômetros rodados (KM driven)",
        min_value=0.0,
        format="%.2f",
        step=100.0,
        help="Digite a quantidade de quilômetros rodados pelo carro. Valores decimais são aceitos."
    )

    ano = st.number_input(
        "Ano do carro",
        min_value=1900,
        max_value=2024,
        step=1,
        help="Digite o ano de fabricação do carro. Deve ser um valor inteiro entre 1900 e 2024."
    )

    # Carregar o modelo salvo
    caminho_modelo = "modelo_random_forest.pkl"  # Ajuste o caminho se necessário
    modelo = carregar_modelo(caminho_modelo)

    if modelo:
        if st.button("Prever Preço"):
            try:
                # Preparar os dados de entrada para o modelo
                entrada = np.array([[km, ano]])
                
                # Fazer a previsão
                preco_previsto = modelo.predict(entrada)

                # Exibir o resultado
                st.success(f"Preço Previsto: **R$ {preco_previsto[0]:,.2f}**")
                st.info(
                    "Observação: Esta previsão é baseada nos dados disponíveis e pode variar dependendo de outros fatores."
                )
            except Exception as e:
                st.error(f"Erro ao realizar a previsão: {e}")
    else:
        st.error("O modelo não pôde ser carregado. Certifique-se de que o arquivo do modelo está disponível.")

    # Rodapé com informações adicionais
    st.markdown(
        """
        ---
        **Desenvolvido por [Seu Nome/Empresa]**  
        Este modelo foi treinado utilizando dados históricos de veículos.
        """
    )

if __name__ == "__main__":
    main()

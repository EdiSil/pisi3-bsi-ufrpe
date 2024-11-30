import streamlit as st
import numpy as np
import requests
import pickle

# Função para baixar e carregar o modelo Random Forest
@st.cache_resource
def carregar_modelo_via_url(url):
    """
    Faz o download e carrega o modelo Random Forest a partir de uma URL.

    Args:
        url (str): URL do arquivo do modelo.

    Returns:
        object: Modelo treinado.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        modelo = pickle.loads(response.content)
        return modelo
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Função principal da aplicação
def main():
    st.title("Previsão de Preço de Carros 🚗")
    st.markdown(
        """
        Esta aplicação utiliza um modelo Random Forest para prever o preço de carros com base em:
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

    # URL do modelo Random Forest
    modelo_url = "https://github.com/EdiSil/pisi3-bsi-ufrpe/pages/5_Classificação Random Forest.py"

    # Carregar o modelo
    modelo = carregar_modelo_via_url(modelo_url)

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
        st.error("Não foi possível carregar o modelo. Verifique a URL ou o formato do arquivo.")

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

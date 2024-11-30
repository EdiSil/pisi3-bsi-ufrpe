import streamlit as st

st.set_page_config(
    page_title = "PISI3 - BSI - UFRPE por Edniz Silva",
    layout = "wide",
    menu_items = {
        'About': '''Este projeto foi desenvolvido com fins acadêmicos, utilizando técnicas de Machine Learning (ML) e Knowledge Discovery 
        in Databases (KDD), com o objetivo de extrair conhecimento valioso a partir de dados brutos e apoiar decisões de negócios, 
        como ajustar preços de venda ou identificar modelos mais populares com base nas características.
        '''
    }
)

st.markdown(f'''
    <h1>UFRPE PISI3</h1>
    <br>
Este projeto foi desenvolvido com fins acadêmicos, utilizando técnicas de Machine Learning (ML) e Knowledge Discovery 
in Databases (KDD), com o objetivo de extrair conhecimento valioso a partir de dados brutos e apoiar decisões de negócios, 
como ajustar preços de venda ou identificar modelos mais populares com base nas características. do curso de Bacharelado em Sistemas de
Informação (BSI) da Sede da Universidade Federal Rural de Pernambuco (UFRPE).
    <br>
    Alguns dos exemplos são:
    <ul>
            <li>Páginas e componentes do Streamlit.</li>
            <li>Uso do Pandas.</li>                    
            <li>Visualização de dados.</li>
            <li>Aprendizado de Máquina: Agrupamento e Classificação.</li>
    </ul>
    Contato: ednizsilva@ufrpe.br<br>    
''', unsafe_allow_html=True)

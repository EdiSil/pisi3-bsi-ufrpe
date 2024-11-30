import streamlit as st

# Interface para predizer preço baseado em novas entradas
st.title("Previsão de Preço de Carros")
km = st.number_input("KM driven")
ano = st.number_input("Ano do Carro")

if st.button("Prever Preço"):
    prediction = model.predict([[km, ano]])
    st.write(f"Preço Previsto: R$ {prediction[0]:.2f}")

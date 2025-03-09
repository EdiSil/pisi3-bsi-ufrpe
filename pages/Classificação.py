import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class SistemaClassificacaoCarros:
    def __init__(self):
        self.dados = None
        self.modelo = None
        self.codificadores = {}
        self.normalizador = StandardScaler()
        self.coluna_alvo = 'faixa_preco'
        
    def carregar_dados(self, caminho_arquivo):
        """Carrega e pré-processa o conjunto de dados de carros"""
        self.dados = pd.read_csv(caminho_arquivo)
        # Criar faixas de preço para classificação
        faixas_preco = [0, 200000, 300000, 400000, float('inf')]
        rotulos_preco = ['Econômico', 'Intermediário', 'Premium', 'Luxo']
        self.dados['faixa_preco'] = pd.cut(self.dados['preco'], 
                                        bins=faixas_preco, 
                                        labels=rotulos_preco)
        return self.dados
    
    def preprocessar_dados(self):
        """Prepara os dados para treinamento do modelo"""
        # Selecionar características para classificação
        caracteristicas = ['marca', 'modelo', 'ano', 'quilometragem', 'combustivel',
                        'car_documents', 'tipo', 'transmissão']
        
        # Codificar variáveis categóricas
        X = self.dados[caracteristicas].copy()
        for coluna in X.select_dtypes(include=['object']):
            self.codificadores[coluna] = LabelEncoder()
            X[coluna] = self.codificadores[coluna].fit_transform(X[coluna])
        
        # Normalizar características numéricas
        caracteristicas_numericas = ['ano', 'quilometragem']
        X[caracteristicas_numericas] = self.normalizador.fit_transform(X[caracteristicas_numericas])
        
        y = self.codificadores[self.coluna_alvo] = LabelEncoder()
        y = y.fit_transform(self.dados[self.coluna_alvo])
        
        return X, y
    
    def treinar_modelo(self, X, y):
        """Treina o classificador Random Forest"""
        self.modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        self.modelo.fit(X, y)
        return self.modelo
    
    def prever(self, dados_entrada):
        """Realiza previsões para novos dados"""
        # Pré-processar dados de entrada
        for coluna in dados_entrada.columns:
            if coluna in self.codificadores:
                dados_entrada[coluna] = self.codificadores[coluna].transform(dados_entrada[coluna])
        
        # Normalizar características numéricas
        caracteristicas_numericas = ['ano', 'quilometragem']
        dados_entrada[caracteristicas_numericas] = self.normalizador.transform(dados_entrada[caracteristicas_numericas])
        
        # Fazer previsão
        previsao = self.modelo.predict(dados_entrada)
        return self.codificadores[self.coluna_alvo].inverse_transform(previsao)

def main():
    st.set_page_config(page_title="Sistema de Classificação de Preços de Carros", layout="wide")
    st.title("Sistema de Classificação de Preços")
    
    # Inicializar o sistema de classificação
    sistema = SistemaClassificacaoCarros()
    
    # Carregar dados
    dados = sistema.carregar_dados('Datas/1_Cars_processado.csv')
    
    # Treinar modelo
    X, y = sistema.preprocessar_dados()
    modelo = sistema.treinar_modelo(X, y)
    
    # Interface do usuário
    st.sidebar.header("Previsão de Faixa de Preço")
    marca = st.sidebar.selectbox("Marca", dados['marca'].unique())
    modelo = st.sidebar.selectbox("Modelo", dados[dados['marca'] == marca]['modelo'].unique())
    ano = st.sidebar.number_input("Ano", min_value=2000, max_value=2023, value=2020)
    quilometragem = st.sidebar.number_input("Quilometragem", min_value=0, value=50000)
    combustivel = st.sidebar.selectbox("Tipo de Combustível", dados['combustivel'].unique())
    car_documents = st.sidebar.selectbox("Documentação", dados['car_documents'].unique())
    tipo = st.sidebar.selectbox("Tipo", dados['tipo'].unique())
    transmissao = st.sidebar.selectbox("Transmissão", dados['transmissão'].unique())
    
    if st.sidebar.button("Prever Faixa de Preço", use_container_width=True):
        dados_entrada = pd.DataFrame({
            'marca': [marca],
            'modelo': [modelo],
            'ano': [ano],
            'quilometragem': [quilometragem],
            'combustivel': [combustivel],
            'car_documents': [car_documents],
            'tipo': [tipo],
            'transmissão': [transmissao]
        })
        
        previsao = sistema.prever(dados_entrada)
        st.sidebar.success(f"Faixa de Preço Prevista: {previsao[0]}")
    
    # Área de visualização
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição das Faixas de Preço")
        fig, ax = plt.subplots(figsize=(10, 6))
        dados['faixa_preco'].value_counts().plot(kind='bar')
        plt.title('Distribuição das Faixas de Preço dos Carros')
        plt.xlabel('Faixa de Preço')
        plt.ylabel('Quantidade')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Matriz de Confusão do Modelo")
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = modelo.predict(X_teste)
        cm = confusion_matrix(y_teste, y_pred)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                    xticklabels=sistema.codificadores[sistema.coluna_alvo].classes_,
                    yticklabels=sistema.codificadores[sistema.coluna_alvo].classes_)
        plt.title('Matriz de Confusão')
        st.pyplot(fig)

if __name__ == "__main__":
    main()

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
        faixas_preco = [0, 200.000, 300.000, 400.000, float('inf')]
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
        faixa_preco = self.codificadores[self.coluna_alvo].inverse_transform(previsao)[0]
        
        # Calcular valor estimado (média da faixa)
        valor_estimado = self.dados[self.dados['faixa_preco'] == faixa_preco]['preco'].mean()
        
        return faixa_preco, valor_estimado

def main():
    st.set_page_config(page_title="Sistema de Classificação de Preços de Carros", layout="wide")
    st.title("Sistema de Classificação de Preços")
    
    # Inicializar o sistema de classificação
    sistema = SistemaClassificacaoCarros()
    
    # Carregar dados
    dados = sistema.carregar_dados('Datas/1_Cars_processado.csv')
    
    # Treinar modelo
    X, y = sistema.preprocessar_dados()
    sistema.treinar_modelo(X, y)
    
    # Interface do usuário
    st.sidebar.header("Previsão de Faixa de Preço")
    marca = st.sidebar.selectbox("Marca", dados['marca'].unique())
    modelo = st.sidebar.selectbox("Modelo", dados[dados['marca'] == marca]['modelo'].unique())
    ano = st.sidebar.number_input("Ano", min_value=2000, max_value=2023, value=2020)
    quilometragem = st.sidebar.number_input("Quilometragem", min_value=0, value=50000)
    combustivel = st.sidebar.selectbox("Tipo de Combustível", dados['combustivel'].unique())
    tipo = st.sidebar.selectbox("Tipo", dados['tipo'].unique())
    transmissao = st.sidebar.selectbox("Transmissão", dados['transmissão'].unique())
    
    # Área de visualização
    st.subheader("DISTRIBUIÇÃO DAS FAIXAS DE PREÇO")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    dados['faixa_preco'].value_counts().plot(kind='bar')
    plt.title('DISTRIBUIÇÃO DAS FAIXAS DE PREÇO DOS CARROS', fontsize=12, pad=20, color='black')
    plt.xlabel('FAIXA DE PREÇO', fontsize=10, color='black')
    plt.ylabel('QUANTIDADE', fontsize=10, color='black')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)
    
    st.subheader("MATRIZ DE CONFUSÃO DO MODELO")
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = sistema.modelo.predict(X_teste)
    cm = confusion_matrix(y_teste, y_pred)
    
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    vmin = cm.min().min()
    vmax = cm.max().max()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax2, cmap='YlOrRd',
                xticklabels=sistema.codificadores[sistema.coluna_alvo].classes_,
                yticklabels=sistema.codificadores[sistema.coluna_alvo].classes_,
                vmin=vmin, vmax=vmax)
    plt.title('MATRIZ DE CONFUSÃO', fontsize=12, pad=20, color='black')
    plt.xlabel('PREVISÃO', fontsize=10, color='black')
    plt.ylabel('VALOR REAL', fontsize=10, color='black')
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Área de previsão com destaque
    st.subheader("PREVISÃO DE FAIXA DE PREÇO")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("REALIZAR PREVISÃO", use_container_width=True):
            dados_entrada = pd.DataFrame({
                'marca': [marca],
                'modelo': [modelo],
                'ano': [ano],
                'quilometragem': [quilometragem],
                'combustivel': [combustivel],
                'car_documents': ['Original'],
                'tipo': [tipo],
                'transmissão': [transmissao]
            })
            
            previsao, valor_estimado = sistema.prever(dados_entrada)
            with col2:
                st.success(
                    f"Faixa de Preço Prevista: {previsao}\n\n" +
                    f"Valor Estimado: R$ {valor_estimado:,.2f}",
                    icon="✨"
                )

if __name__ == "__main__":
    main()

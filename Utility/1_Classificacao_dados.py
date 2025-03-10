import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

class ClassificadorCarros:
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
        faixa_preco = self.codificadores[self.coluna_alvo].inverse_transform(previsao)
        
        # Calcular valor estimado (média da faixa)
        valores_estimados = []
        for faixa in faixa_preco:
            valor_estimado = self.dados[self.dados['faixa_preco'] == faixa]['preco'].mean()
            valores_estimados.append(valor_estimado)
        
        return faixa_preco, valores_estimados

def main():
    # Definir caminhos dos arquivos
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    arquivo_entrada = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1_Cars_processado.csv")
    arquivo_saida = os.path.join(desktop_path, "Cars_classificacao.csv")
    
    # Inicializar o classificador
    classificador = ClassificadorCarros()
    
    try:
        # Carregar e preparar dados
        print("Carregando dados...")
        dados = classificador.carregar_dados(arquivo_entrada)
        
        # Treinar modelo
        print("Preparando dados e treinando modelo...")
        X, y = classificador.preprocessar_dados()
        classificador.treinar_modelo(X, y)
        
        # Preparar dados para previsão
        dados_predicao = dados[['marca', 'modelo', 'ano', 'quilometragem', 'combustivel',
                               'car_documents', 'tipo', 'transmissão']].copy()
        
        # Realizar previsões
        print("Realizando previsões...")
        faixas_previstas, valores_estimados = classificador.prever(dados_predicao)
        
        # Preparar resultados
        resultados = dados.copy()
        resultados['faixa_preco_prevista'] = faixas_previstas
        resultados['valor_estimado'] = valores_estimados
        
        # Salvar resultados
        print(f"Salvando resultados em {arquivo_saida}...")
        resultados.to_csv(arquivo_saida, index=False)
        print("Processo concluído com sucesso!")
        
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")

if __name__ == "__main__":
    main()

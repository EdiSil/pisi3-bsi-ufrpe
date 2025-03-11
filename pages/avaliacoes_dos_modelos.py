import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

class AvaliacaoModelos:
    def __init__(self, caminho_arquivo, coluna_alvo):
        """Inicializa o pipeline de avaliação de modelos."""
        self.caminho_arquivo = caminho_arquivo
        self.coluna_alvo = coluna_alvo
        self.dados = None
        self.caracteristicas = None
        self.alvo = None
        self.resultados = []
        self.modelos = {}

    def carregar_dados(self):
        """Carrega o conjunto de dados."""
        try:
            self.dados = pd.read_csv(self.caminho_arquivo)
            st.success("Dados carregados com sucesso.")
            return True
        except FileNotFoundError:
            st.error(f"Erro: O arquivo {self.caminho_arquivo} não foi encontrado.")
            return False
        except Exception as e:
            st.error(f"Erro ao carregar os dados: {e}")
            return False

    def preparar_dados(self, colunas_caracteristicas):
        """Prepara os dados para treinamento e teste."""
        if self.dados is not None:
            # Criar uma cópia dos dados para evitar modificar o original
            self.caracteristicas = self.dados[colunas_caracteristicas].copy()
            
            # Converter colunas categóricas para numéricas usando codificação one-hot
            colunas_categoricas = self.caracteristicas.select_dtypes(include=['object']).columns
            if not colunas_categoricas.empty:
                self.caracteristicas = pd.get_dummies(self.caracteristicas, columns=colunas_categoricas)
            
            # Tratar valores ausentes
            self.caracteristicas = self.caracteristicas.fillna(self.caracteristicas.mean())
            
            # Garantir que todos os dados sejam numéricos
            self.caracteristicas = self.caracteristicas.astype(float)
            
            # Preparar variável alvo
            self.alvo = self.dados[self.coluna_alvo].astype(int)
            
            st.success("Dados preparados para treinamento e teste.")
            return True
        else:
            st.error("Erro: Os dados não foram carregados.")
            return False

    def treinar_avaliar(self, modelo, nome_modelo):
        """Treina e avalia um modelo específico."""
        if self.caracteristicas is not None and self.alvo is not None:
            X_treino, X_teste, y_treino, y_teste = train_test_split(
                self.caracteristicas, self.alvo, test_size=0.3, random_state=42
            )

            # Treinamento do modelo
            modelo.fit(X_treino, y_treino)
            previsoes = modelo.predict(X_teste)
            self.modelos[nome_modelo] = modelo

            # Avaliação do modelo
            relatorio = classification_report(y_teste, previsoes, output_dict=True)
            acuracia = accuracy_score(y_teste, previsoes)
            matriz_confusao = confusion_matrix(y_teste, previsoes)

            self.resultados.append({
                'Modelo': nome_modelo,
                'Metricas': relatorio,
                'Acuracia': acuracia,
                'Matriz_Confusao': matriz_confusao,
                'Dados_Teste': (X_teste, y_teste)
            })
            st.success(f"Modelo {nome_modelo} avaliado com sucesso.")
            return True
        else:
            st.error("Erro: Os dados não foram preparados para treinamento.")
            return False

    def plotar_matriz_confusao(self, nome_modelo):
        """Plota a matriz de confusão para um modelo específico."""
        for resultado in self.resultados:
            if resultado['Modelo'] == nome_modelo:
                plt.figure(figsize=(8, 6))
                mc = resultado['Matriz_Confusao']
                vmin = 0
                vmax = mc.max().max()
                sns.heatmap(mc, annot=True, fmt='d', cmap='YlOrRd', vmin=vmin, vmax=vmax)
                plt.title(f'Matriz de Confusão - {nome_modelo}')
                plt.ylabel('Real')
                plt.xlabel('Previsto')
                st.pyplot(plt)
                plt.close()

    def plotar_importancia_caracteristicas(self, nome_modelo):
        """Plota a importância das características para modelos que suportam esta funcionalidade."""
        if nome_modelo in self.modelos:
            modelo = self.modelos[nome_modelo]
            
            # Encontrar os dados de teste para este modelo
            dados_teste = None
            for resultado in self.resultados:
                if resultado['Modelo'] == nome_modelo:
                    dados_teste = resultado['Dados_Teste']
                    break
            
            if dados_teste:
                X_teste, y_teste = dados_teste
                
                if hasattr(modelo, 'feature_importances_'):
                    # Usar feature_importances_ para modelos que suportam
                    importancias = modelo.feature_importances_
                else:
                    # Usar permutation importance para modelos como SVM
                    result = permutation_importance(
                        modelo, X_teste, y_teste,
                        n_repeats=10,
                        random_state=42
                    )
                    importancias = result.importances_mean
                
                nomes_caracteristicas = self.caracteristicas.columns
                
                plt.figure(figsize=(10, 6))
                sns.barplot(x=importancias, y=nomes_caracteristicas)
                plt.title(f'Importância das Características - {nome_modelo}')
                plt.xlabel('Importância')
                st.pyplot(plt)
                plt.close()
            else:
                # Usar coeficientes para modelos lineares ou outros métodos alternativos
                if hasattr(modelo, 'coef_'):
                    importancias = np.abs(modelo.coef_).mean(axis=0) if len(modelo.coef_.shape) > 1 else np.abs(modelo.coef_)
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x=importancias, y=nomes_caracteristicas)
                    plt.title(f'Importância das Características (Coeficientes) - {nome_modelo}')
                    plt.xlabel('Importância')
                    st.pyplot(plt)
                    plt.close()
                else:
                    st.info(f"O modelo {nome_modelo} não suporta visualização de importância de características.")

    def exibir_metricas(self, nome_modelo):
        """Exibe as métricas detalhadas para um modelo específico."""
        for resultado in self.resultados:
            if resultado['Modelo'] == nome_modelo:
                metricas = resultado['Metricas']
                st.write(f"### Métricas Detalhadas - {nome_modelo}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Acurácia", f"{resultado['Acuracia']:.3f}")
                with col2:
                    st.metric("Precisão Média", f"{metricas['macro avg']['precision']:.3f}")
                with col3:
                    st.metric("Recall Médio", f"{metricas['macro avg']['recall']:.3f}")

def main():
    st.set_page_config(page_title="Avaliação de Modelos de Machine Learning", layout="wide")
    st.title("Sistema de Avaliação dos Modelos de Machine Learning")

    # Instância da classe AvaliacaoModelos
    caminho_arquivo_entrada = os.path.join('Datas', '3_Cars_predictions.csv')
    avaliador = AvaliacaoModelos(caminho_arquivo_entrada, coluna_alvo='Cluster')

    # Carregar e preparar dados
    if avaliador.carregar_dados():
        # Seleção de características
        todas_caracteristicas = avaliador.dados.columns.tolist()
        caracteristicas_selecionadas = st.sidebar.multiselect(
            "Selecione as Características",
            [col for col in todas_caracteristicas if col not in ['car_documents', 'Predicted Price', 'Características']],
            default=['modelo', 'ano', 'preco', 'combustivel', 'tipo', 'transmissão', 'year_range', 'km\'s driven_range']
        )

        if caracteristicas_selecionadas:
            if avaliador.preparar_dados(caracteristicas_selecionadas):
                # Configurações dos modelos
                st.sidebar.header("Configurações dos Modelos")
                
                # Seleção do modelo
                opcoes_modelos = {
                    "SVM": SVC(),
                    "Random Forest": RandomForestClassifier(),
                    "KNN": KNeighborsClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier()
                }
                
                modelo_selecionado = st.sidebar.selectbox(
                    "Selecione o Modelo",
                    list(opcoes_modelos.keys())
                )

                # Treinamento e avaliação do modelo selecionado
                if st.sidebar.button("Treinar e Avaliar Modelo"):
                    with st.spinner(f"Treinando {modelo_selecionado}..."):
                        modelo = opcoes_modelos[modelo_selecionado]
                        avaliador.treinar_avaliar(modelo, modelo_selecionado)

                # Visualização dos resultados
                if avaliador.resultados:
                    st.header(f"Resultados da Avaliação do Modelo {modelo_selecionado}")
                    
                    # Métricas detalhadas
                    avaliador.exibir_metricas(modelo_selecionado)
                    
                    # Matriz de Confusão
                    st.subheader("Matriz de Confusão")
                    avaliador.plotar_matriz_confusao(modelo_selecionado)
                    
                    # Importância das Características
                    st.subheader("Importância das Características")
                    avaliador.plotar_importancia_caracteristicas(modelo_selecionado)
                    
                    # Área de previsão com destaque
                    st.subheader("PREVISÃO DE PREÇO")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        st.markdown("""
                        <style>
                        div.stSuccess {
                            text-align: center;
                            padding: 20px;
                            font-size: 18px;
                            line-height: 1.6;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        st.success(
                            f"""✨ FAIXA DE PREÇO PREVISTA ✨
                            {previsao}
                        
                        💰 VALOR ESTIMADO 💰
                        R$ {valor_estimado:,.2f}""".replace(',', '_').replace('.', ',').replace('_', '.'),
                            icon=None
                        )

def main():
    st.set_page_config(page_title="Avaliação de Modelos de Machine Learning", layout="wide")
    st.title("Sistema de Avaliação dos Modelos de Machine Learning")

    # Instância da classe AvaliacaoModelos
    caminho_arquivo_entrada = os.path.join('Datas', '3_Cars_predictions.csv')
    avaliador = AvaliacaoModelos(caminho_arquivo_entrada, coluna_alvo='Cluster')

    # Carregar e preparar dados
    if avaliador.carregar_dados():
        # Seleção de características
        todas_caracteristicas = avaliador.dados.columns.tolist()
        caracteristicas_selecionadas = st.sidebar.multiselect(
            "Selecione as Características",
            [col for col in todas_caracteristicas if col not in ['car_documents', 'Predicted Price', 'Características']],
            default=['modelo', 'ano', 'preco', 'combustivel', 'tipo', 'transmissão', 'year_range', "km's driven_range", 'marca', 'quilometragem', 'Car Age']
        )

        if caracteristicas_selecionadas:
            if avaliador.preparar_dados(caracteristicas_selecionadas):
                # Configurações dos modelos
                st.sidebar.header("Configurações dos Modelos")
                
                # Seleção do modelo
                opcoes_modelos = {
                    "SVM": SVC(),
                    "Random Forest": RandomForestClassifier(),
                    "KNN": KNeighborsClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier()
                }
                
                modelo_selecionado = st.sidebar.selectbox(
                    "Selecione o Modelo",
                    list(opcoes_modelos.keys())
                )

                # Treinamento e avaliação do modelo selecionado
                if st.sidebar.button("Treinar e Avaliar Modelo"):
                    with st.spinner(f"Treinando {modelo_selecionado}..."):
                        modelo = opcoes_modelos[modelo_selecionado]
                        avaliador.treinar_avaliar(modelo, modelo_selecionado)

                # Visualização dos resultados
                if avaliador.resultados:
                    st.header(f"Resultados da Avaliação do Modelo {modelo_selecionado}")
                    
                    # Métricas detalhadas
                    avaliador.exibir_metricas(modelo_selecionado)
                    
                    # Matriz de Confusão
                    st.subheader("Matriz de Confusão")
                    avaliador.plotar_matriz_confusao(modelo_selecionado)
                    
                    # Importância das Características
                    st.subheader("Importância das Características")
                    avaliador.plotar_importancia_caracteristicas(modelo_selecionado)

if __name__ == "__main__":
    main()

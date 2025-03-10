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

class ModelEvaluation:
    def __init__(self, file_path, target_column):
        """Inicializa o pipeline de avaliação de modelos."""
        self.file_path = file_path
        self.target_column = target_column
        self.data = None
        self.features = None
        self.target = None
        self.results = []
        self.models = {}

    def load_data(self):
        """Carrega o dataset."""
        try:
            self.data = pd.read_csv(self.file_path)
            st.success("Dados carregados com sucesso.")
            return True
        except FileNotFoundError:
            st.error(f"Erro: O arquivo {self.file_path} não foi encontrado.")
            return False
        except Exception as e:
            st.error(f"Erro ao carregar os dados: {e}")
            return False

    def prepare_data(self, feature_columns):
        """Prepara os dados para treinamento e teste."""
        if self.data is not None:
            self.features = self.data[feature_columns]
            self.target = self.data[self.target_column]
            st.success("Dados preparados para treinamento e teste.")
            return True
        else:
            st.error("Erro: Os dados não foram carregados.")
            return False

    def train_evaluate(self, model, model_name):
        """Treina e avalia um modelo específico."""
        if self.features is not None and self.target is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                self.features, self.target, test_size=0.3, random_state=42
            )

            # Treinamento do modelo
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            self.models[model_name] = model

            # Avaliação do modelo
            report = classification_report(y_test, predictions, output_dict=True)
            accuracy = accuracy_score(y_test, predictions)
            conf_matrix = confusion_matrix(y_test, predictions)

            self.results.append({
                'Model': model_name,
                'Metrics': report,
                'Accuracy': accuracy,
                'Confusion_Matrix': conf_matrix,
                'Test_Data': (X_test, y_test)
            })
            st.success(f"Modelo {model_name} avaliado com sucesso.")
            return True
        else:
            st.error("Erro: Os dados não foram preparados para treinamento.")
            return False

    def plot_confusion_matrix(self, model_name):
        """Plota a matriz de confusão para um modelo específico."""
        for result in self.results:
            if result['Model'] == model_name:
                plt.figure(figsize=(8, 6))
                sns.heatmap(result['Confusion_Matrix'], annot=True, fmt='d', cmap='Blues')
                plt.title(f'Matriz de Confusão - {model_name}')
                plt.ylabel('Real')
                plt.xlabel('Predito')
                st.pyplot(plt)
                plt.close()

    def plot_feature_importance(self, model_name):
        """Plota a importância das features para modelos que suportam esta funcionalidade."""
        if model_name in self.models:
            model = self.models[model_name]
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = self.features.columns
                
                plt.figure(figsize=(10, 6))
                sns.barplot(x=importances, y=feature_names)
                plt.title(f'Importância das Features - {model_name}')
                plt.xlabel('Importância')
                st.pyplot(plt)
                plt.close()
            else:
                st.info(f"O modelo {model_name} não suporta visualização de importância de features.")

    def display_metrics(self, model_name):
        """Exibe as métricas detalhadas para um modelo específico."""
        for result in self.results:
            if result['Model'] == model_name:
                metrics = result['Metrics']
                st.write(f"### Métricas Detalhadas - {model_name}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Acurácia", f"{result['Accuracy']:.3f}")
                with col2:
                    st.metric("Precisão Média", f"{metrics['macro avg']['precision']:.3f}")
                with col3:
                    st.metric("Recall Médio", f"{metrics['macro avg']['recall']:.3f}")

def main():
    st.set_page_config(page_title="Avaliação de Modelos de Machine Learning", layout="wide")
    st.title("Sistema de Avaliação de Modelos de Machine Learning")

    # Sidebar para configurações
    st.sidebar.header("Configurações")
    
    # Seleção de arquivo
    input_file = st.sidebar.file_uploader("Carregar arquivo CSV", type=['csv'])
    if input_file is not None:
        input_file_path = input_file.name
    else:
        input_file_path = os.path.join(os.path.expanduser('~'), 'Desktop', '03_Cars_predictions.csv')

    # Instância da classe ModelEvaluation
    evaluator = ModelEvaluation(input_file_path, target_column='Cluster')

    # Carregar e preparar dados
    if evaluator.load_data():
        # Seleção de features
        all_features = evaluator.data.columns.tolist()
        selected_features = st.sidebar.multiselect(
            "Selecionar Features",
            all_features,
            default=['quilometragem', 'Car Age', 'Cluster']
        )

        if selected_features:
            if evaluator.prepare_data(selected_features):
                # Configurações dos modelos
                st.sidebar.header("Configurações dos Modelos")

                # Treinamento e avaliação dos modelos
                if st.sidebar.button("Treinar e Avaliar Modelos"):
                    with st.spinner("Treinando modelos..."):
                        # Treinamento dos diferentes modelos
                        evaluator.train_evaluate(SVC(), "SVM")
                        evaluator.train_evaluate(RandomForestClassifier(), "Random Forest")
                        evaluator.train_evaluate(KNeighborsClassifier(), "KNN")
                        evaluator.train_evaluate(GradientBoostingClassifier(), "Gradient Boosting")

                # Visualização dos resultados
                if evaluator.results:
                    st.header("Visualização dos Resultados")
                    
                    # Seleção do modelo para visualização
                    model_names = [result['Model'] for result in evaluator.results]
                    selected_model = st.selectbox("Selecione o Modelo", model_names)

                    # Exibição das métricas e visualizações
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Matriz de Confusão")
                        evaluator.plot_confusion_matrix(selected_model)
                    with col2:
                        st.subheader("Importância das Features")
                        evaluator.plot_feature_importance(selected_model)

                    # Métricas detalhadas
                    evaluator.display_metrics(selected_model)

if __name__ == "__main__":
    main()

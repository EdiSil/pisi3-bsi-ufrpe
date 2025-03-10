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
            # Create a copy of the data to avoid modifying the original
            self.features = self.data[feature_columns].copy()
            
            # Convert categorical columns to numeric using one-hot encoding
            categorical_columns = self.features.select_dtypes(include=['object']).columns
            if not categorical_columns.empty:
                self.features = pd.get_dummies(self.features, columns=categorical_columns)
            
            # Handle missing values
            self.features = self.features.fillna(self.features.mean())
            
            # Ensure all data is numeric
            self.features = self.features.astype(float)
            
            # Prepare target variable
            self.target = self.data[self.target_column].astype(int)
            
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
                plt.title(f'Importância dos recursos - {model_name}')
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
                
                st.metric("Acurácia", f"{result['Accuracy']:.3f}")
                st.metric("Precisão Média", f"{metrics['macro avg']['precision']:.3f}")
                st.metric("Recall Médio", f"{metrics['macro avg']['recall']:.3f}")

def main():
    st.set_page_config(page_title="Avaliação de Modelos de Machine Learning", layout="wide")
    st.title("Sistema de Avaliação de Modelos de Machine Learning")

    # Instância da classe ModelEvaluation
    input_file_path = os.path.join('Datas', '3_Cars_predictions.csv')
    evaluator = ModelEvaluation(input_file_path, target_column='Cluster')

    # Carregar e preparar dados
    if evaluator.load_data():
        # Seleção de features
        all_features = evaluator.data.columns.tolist()
        selected_features = st.sidebar.multiselect(
            "Selecione os Recursos",
            all_features,
            default=['quilometragem', 'Car Age', 'Cluster']
        )

        if selected_features:
            if evaluator.prepare_data(selected_features):
                # Configurações dos modelos
                st.sidebar.header("Configurações dos Modelos")
                
                # Seleção do modelo
                model_options = {
                    "SVM": SVC(),
                    "Random Forest": RandomForestClassifier(),
                    "KNN": KNeighborsClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier()
                }
                
                selected_model = st.sidebar.selectbox(
                    "Selecione o Modelo",
                    list(model_options.keys())
                )

                # Treinamento e avaliação do modelo selecionado
                if st.sidebar.button("Treinar e Avaliar Modelo"):
                    with st.spinner(f"Treinando {selected_model}..."):
                        model = model_options[selected_model]
                        evaluator.train_evaluate(model, selected_model)

                # Visualização dos resultados
                if evaluator.results:
                    st.header(f"Resultados da Avaliação do Modelo {selected_model}")
                    
                    # Métricas detalhadas
                    evaluator.display_metrics(selected_model)
                    
                    # Matriz de Confusão
                    st.subheader("Matriz de Confusão")
                    evaluator.plot_confusion_matrix(selected_model)
                    
                    # Importância das Features
                    st.subheader("Importância dos recursos")
                    evaluator.plot_feature_importance(selected_model)

                    # Métricas detalhadas
                    evaluator.display_metrics(selected_model)

if __name__ == "__main__":
    main()

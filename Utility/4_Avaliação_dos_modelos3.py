import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import os

class ModelEvaluation:
    def __init__(self, file_path, target_column):
        """Inicializa o pipeline de avaliação de modelos."""
        self.file_path = file_path
        self.target_column = target_column
        self.data = None
        self.features = None
        self.target = None
        self.results = []

    def load_data(self):
        """Carrega o dataset."""
        try:
            self.data = pd.read_csv(self.file_path)
            print("Dados carregados com sucesso.")
        except FileNotFoundError:
            print(f"Erro: O arquivo {self.file_path} não foi encontrado.")
        except Exception as e:
            print(f"Erro ao carregar os dados: {e}")

    def prepare_data(self, feature_columns):
        """Prepara os dados para treinamento e teste."""
        if self.data is not None:
            self.features = self.data[feature_columns]
            self.target = self.data[self.target_column]
            print("Dados preparados para treinamento e teste.")
        else:
            print("Erro: Os dados não foram carregados.")

    def train_evaluate(self, model, model_name, use_smote=False):
        """Treina e avalia um modelo específico."""
        if self.features is not None and self.target is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                self.features, self.target, test_size=0.3, random_state=42
            )

            if use_smote:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"SMOTE aplicado ao modelo {model_name}.")

            # Treinamento do modelo
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Avaliação do modelo
            report = classification_report(y_test, predictions, output_dict=True)
            accuracy = accuracy_score(y_test, predictions)

            self.results.append({
                'Model': model_name,
                'Metrics': report,
                'Accuracy': accuracy
            })
            print(f"Modelo {model_name} avaliado com sucesso.")
        else:
            print("Erro: Os dados não foram preparados para treinamento.")

    def format_results(self):
        """Formata os resultados para exibição e salvamento, com 3 casas decimais."""
        formatted_results = []
        for result in self.results:
            model_name = result['Model']
            metrics = result['Metrics']
            accuracy = result['Accuracy']

            # Formatação com 3 casas decimais
            formatted_results.append({
                'Modelo': model_name,
                'Métrica': 'Precisão',
                'Classe 1': f"{metrics['1']['precision']:.3f}",
                'Classe 2': f"{metrics['2']['precision']:.3f}",
                'Classe 3': f"{metrics['3']['precision']:.3f}",
                'Média': f"{metrics['macro avg']['precision']:.3f}",
                'Média Ponderada': f"{metrics['weighted avg']['precision']:.3f}"
            })
            formatted_results.append({
                'Modelo': model_name,
                'Métrica': 'Recall',
                'Classe 1': f"{metrics['1']['recall']:.3f}",
                'Classe 2': f"{metrics['2']['recall']:.3f}",
                'Classe 3': f"{metrics['3']['recall']:.3f}",
                'Média': f"{metrics['macro avg']['recall']:.3f}",
                'Média Ponderada': f"{metrics['weighted avg']['recall']:.3f}"
            })
            formatted_results.append({
                'Modelo': model_name,
                'Métrica': 'F1-Score',
                'Classe 1': f"{metrics['1']['f1-score']:.3f}",
                'Classe 2': f"{metrics['2']['f1-score']:.3f}",
                'Classe 3': f"{metrics['3']['f1-score']:.3f}",
                'Média': f"{metrics['macro avg']['f1-score']:.3f}",
                'Média Ponderada': f"{metrics['weighted avg']['f1-score']:.3f}"
            })
            formatted_results.append({
                'Modelo': model_name,
                'Métrica': 'Suporte',
                'Classe 1': f"{metrics['1']['support']:.3f}",
                'Classe 2': f"{metrics['2']['support']:.3f}",
                'Classe 3': f"{metrics['3']['support']:.3f}",
                'Média': '-',
                'Média Ponderada': '-'
            })
            formatted_results.append({
                'Modelo': model_name,
                'Métrica': 'Acurácia',
                'Classe 1': '-',
                'Classe 2': '-',
                'Classe 3': '-',
                'Média': f"{accuracy:.3f}",
                'Média Ponderada': '-'
            })
        return formatted_results

    def save_results(self, output_file):
        """Salva os resultados formatados em um arquivo CSV."""
        formatted_results = self.format_results()
        results_df = pd.DataFrame(formatted_results)
        try:
            results_df.to_csv(output_file, index=False)
            print(f"Resultados salvos em: {output_file}")
        except Exception as e:
            print(f"Erro ao salvar os resultados: {e}")

# Caminho do arquivo de entrada e de saída
input_file_path = os.path.join(os.path.expanduser('~'), 'Desktop', '03_Cars_predictions.csv')
output_file_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'Evalucao_Modelos.csv')

# Instância da classe ModelEvaluation
evaluator = ModelEvaluation(input_file_path, target_column='Cluster')

# Execução do pipeline
evaluator.load_data()
features = ['quilometragem', 'Car Age', 'Cluster']  # Seleção de recursos relevantes
evaluator.prepare_data(features)

# Avaliação de modelos
evaluator.train_evaluate(SVC(), "SVM")
evaluator.train_evaluate(SVC(), "SVM + SMOTE", use_smote=True)
evaluator.train_evaluate(RandomForestClassifier(), "Random Forest")
evaluator.train_evaluate(KNeighborsClassifier(), "KNN")

# Salvar resultados
evaluator.save_results(output_file_path)

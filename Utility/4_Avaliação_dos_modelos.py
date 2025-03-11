import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import os

class ModelEvaluation:
    def __init__(self, file_path, target_column):
        self.file_path = file_path
        self.target_column = target_column
        self.data = None
        self.features = None
        self.target = None
        self.results = []

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print("Dados carregados com sucesso.")
        except FileNotFoundError:
            print(f"Erro: O arquivo {self.file_path} não foi encontrado.")
        except Exception as e:
            print(f"Erro ao carregar os dados: {e}")

    def prepare_data(self, feature_columns):
        if self.data is not None:
            self.features = self.data[feature_columns]
            self.target = self.data[self.target_column]
            print("Dados preparados para treinamento e teste.")
        else:
            print("Erro: Os dados não foram carregados.")

    def train_evaluate(self, model, model_name, use_smote=False):
        if self.features is not None and self.target is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                self.features, self.target, test_size=0.3, random_state=42
            )

            if use_smote:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"SMOTE aplicado ao modelo {model_name}.")

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

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

    def get_metric_value(self, metrics, class_label, metric_name):
        return f"{metrics.get(str(class_label), {}).get(metric_name, 0):.3f}"

    def format_results(self):
        formatted_results = []
        for result in self.results:
            model_name = result['Model']
            metrics = result['Metrics']
            accuracy = result['Accuracy']

            for metric in ['precision', 'recall', 'f1-score', 'support']:
                formatted_results.append({
                    'Modelo': model_name,
                    'Métrica': metric.capitalize(),
                    'Closter1': self.get_metric_value(metrics, 1, metric),
                    'Closter2': self.get_metric_value(metrics, 2, metric),
                    'Closter3': self.get_metric_value(metrics, 3, metric),
                    'Closter4': self.get_metric_value(metrics, 4, metric),
                    'Closter5': self.get_metric_value(metrics, 5, metric),
                    'Média': f"{metrics['macro avg'][metric]:.3f}",
                    'Média Ponderada': f"{metrics['weighted avg'][metric]:.3f}" if metric != 'support' else '-'
                })
            formatted_results.append({
                'Modelo': model_name,
                'Métrica': 'Acurácia',
                'Closter1': '-', 'Closter2': '-', 'Closter3': '-', 'Closter4': '-', 'Closter5': '-',
                'Média': f"{accuracy:.3f}",
                'Média Ponderada': '-'
            })
        return formatted_results

    def save_results(self, output_file):
        formatted_results = self.format_results()
        results_df = pd.DataFrame(formatted_results)
        try:
            results_df.to_csv(output_file, index=False)
            print(f"Resultados salvos em: {output_file}")
        except Exception as e:
            print(f"Erro ao salvar os resultados: {e}")

input_file_path = os.path.join(os.path.expanduser('~'), 'Desktop', '3_Cars_predictions.csv')
output_file_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'Evalucao_Modelos.csv')

evaluator = ModelEvaluation(input_file_path, target_column='Cluster')

evaluator.load_data()
features = ['quilometragem', 'Car Age', 'Cluster']
evaluator.prepare_data(features)

evaluator.train_evaluate(SVC(), "SVM")
evaluator.train_evaluate(RandomForestClassifier(), "Random Forest")
evaluator.train_evaluate(KNeighborsClassifier(), "KNN")
evaluator.train_evaluate(GradientBoostingClassifier(), "Gradient Boosting")
evaluator.train_evaluate(DecisionTreeClassifier(), "Decision Tree")

evaluator.save_results(output_file_path)

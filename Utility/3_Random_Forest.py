import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

class CarPricePredictor:
    def __init__(self, file_path):
        """Inicializa o preditor de preços de carros."""
        self.file_path = file_path
        self.data = None
        self.model = None
        self.features = None
        self.target = None

    def load_data(self):
        """Carrega o dataset clusterizado."""
        try:
            self.data = pd.read_csv(self.file_path)
            print("Dados carregados com sucesso.")
        except FileNotFoundError:
            print(f"Erro: O arquivo {self.file_path} não foi encontrado.")
        except Exception as e:
            print(f"Erro ao carregar os dados: {e}")

    def prepare_data(self, feature_columns, target_column):
        """Prepara os dados para o treinamento do modelo."""
        if self.data is not None:
            self.features = self.data[feature_columns]
            self.target = self.data[target_column]
            print("Dados preparados para treinamento.")
        else:
            print("Erro: Os dados não foram carregados.")

    def train_model(self, test_size=0.3, random_state=42):
        """Treina o modelo Random Forest para previsão de preços."""
        if self.features is not None and self.target is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                self.features, self.target, test_size=test_size, random_state=random_state
            )

            # Treinamento do modelo
            self.model = RandomForestRegressor(random_state=random_state)
            self.model.fit(X_train, y_train)
            print("Modelo Random Forest treinado com sucesso.")

            # Avaliação do modelo
            predictions = self.model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            print(f"Erro Quadrático Médio (MSE): {mse}")
            print(f"Coeficiente de Determinação (R2): {r2}")
        else:
            print("Erro: Os dados de treino não foram preparados.")

    def save_predictions(self, output_file):
        """Gera previsões e salva os dados atualizados no local especificado."""
        if self.model is not None and self.features is not None:
            self.data['Predicted Price'] = self.model.predict(self.features)
            try:
                self.data.to_csv(output_file, index=False)
                print(f"Dados com previsões salvos em: {output_file}")
            except Exception as e:
                print(f"Erro ao salvar os dados: {e}")
        else:
            print("Erro: O modelo não está treinado ou os dados não estão disponíveis.")

# Caminho do arquivo clusterizado e de saída
input_file_path = os.path.join(os.path.expanduser('~'), 'Desktop', '02_Cars_dataset_clusterizado.csv')
output_file_path = os.path.join(os.path.expanduser('~'), 'Desktop', '03_Cars_predictions.csv')

# Instância da classe CarPricePredictor
predictor = CarPricePredictor(input_file_path)

# Execução do pipeline de previsão
predictor.load_data()
features = ['quilometragem', 'Car Age', 'Cluster']  # Seleção de recursos relevantes
predictor.prepare_data(features, target_column='preco')
predictor.train_model()
predictor.save_predictions(output_file_path)

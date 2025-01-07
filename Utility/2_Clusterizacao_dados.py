import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

class CarDataClusterer:
    def __init__(self, file_path):
        """Inicializa o clusterizador de dados de carros."""
        self.file_path = file_path
        self.data = None
        self.clustered_data = None

    def load_data(self):
        """Carrega o dataset processado."""
        try:
            self.data = pd.read_csv(self.file_path)
            print("Dados carregados com sucesso.")
        except FileNotFoundError:
            print(f"Erro: O arquivo {self.file_path} não foi encontrado.")
        except Exception as e:
            print(f"Erro ao carregar os dados: {e}")

    def add_car_age(self, current_year):
        """Adiciona a coluna de idade do carro."""
        if self.data is not None:
            self.data['Car Age'] = current_year - self.data['ano']
            print("Coluna 'Car Age' adicionada aos dados.")
        else:
            print("Erro: Os dados não foram carregados.")

    def preprocess_for_clustering(self, feature_columns):
        """Pré-processa os dados para clusterização."""
        if self.data is not None:
            # Selecionar as colunas de interesse
            features = self.data[feature_columns]

            # Normalizar os dados
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            print("Dados normalizados para clusterização.")
            return scaled_features
        else:
            print("Erro: Os dados não foram carregados.")
            return None

    def perform_clustering(self, scaled_features, n_clusters):
        """Executa o algoritmo de clusterização K-Means."""
        if scaled_features is not None:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.data['Cluster'] = kmeans.fit_predict(scaled_features)
            print(f"Clusterização concluída com {n_clusters} clusters.")
        else:
            print("Erro: Os dados normalizados não estão disponíveis.")

    def save_clustered_data(self, output_file):
        """Salva o dataset com os clusters no local especificado."""
        if self.data is not None:
            try:
                self.data.to_csv(output_file, index=False)
                print(f"Dados clusterizados salvos em: {output_file}")
            except Exception as e:
                print(f"Erro ao salvar os dados: {e}")
        else:
            print("Erro: Os dados clusterizados não estão disponíveis para salvar.")

# Caminho do arquivo processado e de saída
input_file_path = os.path.join(os.path.expanduser('~'), 'Desktop', '01_Cars_dataset_processado.csv')
output_file_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'OLX_cars_dataset_clustered.csv')

# Instância da classe CarDataClusterer
clusterer = CarDataClusterer(input_file_path)

# Execução do pipeline de clusterização
current_year = 2025  # Definir o ano atual
clusterer.load_data()
clusterer.add_car_age(current_year)

features_to_cluster = ['quilometragem', 'preco', 'Car Age']  # Selecionar recursos relevantes
scaled_data = clusterer.preprocess_for_clustering(features_to_cluster)
clusterer.perform_clustering(scaled_data, n_clusters=5)
clusterer.save_clustered_data(output_file_path)

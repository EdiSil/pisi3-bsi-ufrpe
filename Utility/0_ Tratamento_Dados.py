import pandas as pd

class DataOverview:
    """
    Classe para realizar uma visão geral do conjunto de dados.
    Inclui informações como: dimensões, valores nulos, tipos de dados,
    estatísticas descritivas, linhas duplicadas e informações detalhadas dos recursos.
    """
    def __init__(self, file_path):
        """
        Inicializa a classe com o caminho do arquivo e carrega o DataFrame.
        :param file_path: Caminho do arquivo CSV.
        """
        self.file_path = file_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """
        Carrega os dados do arquivo CSV para um DataFrame.
        """
        try:
            self.df = pd.read_csv(self.file_path)
            print("\n[INFO] Dados carregados com sucesso!\n")
        except FileNotFoundError:
            print("\n[ERRO] Arquivo não encontrado. Verifique o caminho do arquivo fornecido.\n")
        except Exception as e:
            print(f"\n[ERRO] Ocorreu um erro ao carregar os dados: {e}\n")

    def dataset_shape(self):
        """
        Exibe o número de linhas e colunas do conjunto de dados.
        """
        rows, columns = self.df.shape
        print(f"Number of Rows: {rows}")
        print(f"Number of Columns: {columns}\n")

    def data_head(self, n=2):
        """
        Exibe as primeiras N linhas do conjunto de dados.
        :param n: Número de linhas a serem exibidas (padrão = 2)
        """
        print("\nData Sample:\n")
        print(self.df.head(n))

    def data_info(self):
        """
        Exibe informações sobre tipos de dados e valores não nulos.
        """
        print("\nData Info:\n")
        self.df.info()

    def data_statistics(self):
        """
        Exibe estatísticas descritivas do conjunto de dados.
        """
        print("\nDescriptive Statistics:\n")
        print(self.df.describe().T)

    def duplicated_rows(self):
        """
        Exibe o total de linhas duplicadas no conjunto de dados.
        """
        total_duplicates = self.df.duplicated().sum()
        print("\nDuplicated Rows:")
        print(f"Total: {total_duplicates}\n")

    def detailed_info(self):
        """
        Gera um resumo detalhado do conjunto de dados, incluindo valores ausentes,
        porcentagem de valores ausentes, valores únicos e tipos de dados.
        """
        print("\nDetailed Dataset Overview:\n")
        basic_info = pd.DataFrame({
            "Features": self.df.columns,
            "Missing Values": self.df.isnull().sum().values,
            "Missing Values %": (self.df.isnull().sum().values / len(self.df)) * 100,
            "Unique Values": self.df.nunique().values,
            "Data Types": self.df.dtypes.values
        })
        print(basic_info.reset_index(drop=True))

    def full_overview(self):
        """
        Executa todas as análises de forma sequencial.
        """
        print("========= Visão Geral do Conjunto de Dados =========")
        self.dataset_shape()
        self.data_head()
        self.data_info()
        self.data_statistics()
        self.duplicated_rows()
        self.detailed_info()
        print("\n========= Fim da Análise =========\n")

class DataPreprocessing:
    """
    Classe para realizar o pré-processamento do conjunto de dados.
    Inclui remoção de duplicatas, colunas desnecessárias e linhas discrepantes.
    """
    def __init__(self, df):
        """
        Inicializa a classe com um DataFrame.
        :param df: DataFrame original.
        """
        self.df = df

    def remove_duplicates(self):
        """
        Remove linhas duplicadas do DataFrame.
        """
        print("\n[INFO] Removendo linhas duplicadas...")
        self.df.drop_duplicates(inplace=True)
        print("Linhas duplicadas removidas.\n")

    def remove_unnecessary_columns(self, columns):
        """
        Remove colunas desnecessárias do DataFrame.
        :param columns: Lista de colunas a serem removidas.
        """
        print("[INFO] Removendo colunas desnecessárias...")
        self.df.drop(columns=columns, inplace=True)
        print("Colunas removidas.\n")

    def remove_outliers(self):
        """
        Remove linhas discrepantes com base em condições predefinidas.
        """
        print("[INFO] Removendo linhas discrepantes...")
        self.df = self.df[(self.df["Model"] != "Civic VTi") &
                          (self.df["Model"] != "Civic EXi") &
                          (self.df["Model"] != "Civic VTi Oriel") &
                          (self.df["Model"] != "Cervo") &
                          (self.df["Model"] != "Every Wagon") &
                          (self.df["Model"] != "Liana") &
                          (self.df["Model"] != "Mehran VX") &
                          (self.df["Model"] != "Khyber") &
                          (self.df["Model"] != "Cultus VXL") &
                          (self.df["Model"] != "Corolla Assista") &
                          (self.df["Model"] != "Corolla Axio") &
                          (self.df["Model"] != "Surf") &
                          (self.df["Model"] != "Prius") &
                          (self.df["Model"] != "ISIS")]
        self.df = self.df[self.df["Year"] != 2024]
        print("Linhas discrepantes removidas.\n")

    def preprocess(self):
        """
        Executa todo o pré-processamento.
        """
        self.remove_duplicates()
        self.remove_unnecessary_columns(["Ad ID", "Car Name", "Condition", "Seller Location",
                                         "Registration city", "Description", "Car Features",
                                         "Images URL's", "Car Profile"])
        self.remove_outliers()
        print("[INFO] Pré-processamento concluído!\n")
        return self.df

class DataDiscretization:
    """
    Classe para realizar a discretização de dados em compartimentos.
    """
    def __init__(self, df):
        """
        Inicializa a classe com um DataFrame.
        :param df: DataFrame original.
        """
        self.df = df.copy()

    def discretize_year(self):
        """
        Cria uma nova coluna discretizada para a coluna 'Year'.
        """
        bins = [1999, 2004, 2008, 2012, 2016, 2020, 2024]
        labels = [1, 2, 3, 4, 5, 6]
        self.df["Year_Range"] = pd.cut(self.df["Year"], bins=bins, labels=labels)

    def discretize_km_driven(self):
        """
        Cria uma nova coluna discretizada para a coluna 'KM's driven'.
        """
        bins = [0, 30000, 60000, 90000, 120000, 150000, 180000, 210000,
                224000, 227000, 300000, 330000, 360000, 390000, 410000, 440000,
                470000, 500000, 533530]
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        self.df["KM's driven_Range"] = pd.cut(self.df["KM's driven"], bins=bins, labels=labels)

    def convert_to_integer(self):
        """
        Converte colunas discretizadas para o tipo inteiro.
        """
        cols = ["Year_Range", "KM's driven_Range"]
        for col in cols:
            self.df[col] = self.df[col].astype("int32")

    def check_correlation(self):
        """
        Verifica as correlações entre colunas numéricas e 'Preço'.
        """
        print("\n[INFO] Correlações:\n")
        print(self.df.select_dtypes(["int", "float"]).corr())

    def save_to_csv(self, output_path):
        """
        Salva o DataFrame transformado em um arquivo CSV.
        :param output_path: Caminho de saída do arquivo.
        """
        self.df.to_csv(output_path, index=False)
        print(f"[INFO] Dados salvos em: {output_path}\n")

    def discretize(self, output_path):
        """
        Executa todo o processo de discretização.
        """
        self.discretize_year()
        self.discretize_km_driven()
        self.convert_to_integer()
        self.check_correlation()
        self.save_to_csv(output_path)
        print("[INFO] Discretização concluída!\n")

if __name__ == "__main__":
    # Caminho para o arquivo CSV (Altere conforme o local do seu arquivo)
    file_path = r"C:\Users\Tutu\Desktop\Facul\Projeto 3\Projeto 2024.2\data\OLX_cars_dataset00.csv"
    output_path = r"C:\Users\Tutu\Desktop\Facul\Projeto 3\Projeto 2024.2\data\01_Cars_dataset_processado.csv"

    # Etapa 1: Visão geral dos dados
    overview = DataOverview(file_path)
    overview.full_overview()

    # Etapa 2: Pré-processamento dos dados
    preprocessor = DataPreprocessing(overview.df)
    cleaned_df = preprocessor.preprocess()

    # Etapa 3: Discretização dos dados
    discretizer = DataDiscretization(cleaned_df)
    discretizer.discretize(output_path)

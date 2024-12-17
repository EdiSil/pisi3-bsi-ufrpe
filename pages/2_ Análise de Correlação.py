import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

# Classe principal para a aplicacao
class DataAnalysisApp:
    def __init__(self):
        self.data = None

    def load_data(self, file_path):
        """Carrega os dados do arquivo CSV."""
        try:
            self.data = pd.read_csv(file_path)
            return True
        except Exception as e:
            st.error(f"Erro ao carregar os dados: {e}")
            return False

    def display_data(self):
        """Exibe as primeiras linhas dos dados."""
        if self.data is not None:
            st.subheader("Dados Carregados")
            st.dataframe(self.data.head())

    def plot_correlation_matrix(self):
        """Calcula e exibe a matriz de correlação como um heatmap."""
        if self.data is not None:
            st.subheader("Matriz de Correlação")
            try:
                # Calcular a matriz de correlação
                correlation_matrix = self.data.corr()
                
                # Configurar o tamanho da figura
                plt.figure(figsize=(12, 8))
                
                # Criar o heatmap
                sns.heatmap(
                    correlation_matrix, 
                    annot=True, 
                    fmt=".2f", 
                    cmap='coolwarm', 
                    square=True, 
                    cbar_kws={"shrink": .8}
                )
                plt.title('Matriz de Correlação', fontsize=16)
                plt.xticks(rotation=45)
                plt.yticks(rotation=45)
                
                # Mostrar o gráfico
                st.pyplot(plt)
            except Exception as e:
                st.error(f"Erro ao gerar a matriz de correlação: {e}")

    def perform_anova(self, categorical_var, numerical_var):
        """Realiza o teste ANOVA entre variáveis categóricas e numéricas."""
        try:
            # Agrupar os dados pela variável categórica
            groups = [group[numerical_var].dropna().values for _, group in self.data.groupby(categorical_var)]
            
            # Realizar o teste ANOVA
            f_statistic, p_value = stats.f_oneway(*groups)
            return f_statistic, p_value
        except Exception as e:
            st.error(f"Erro ao realizar ANOVA: {e}")
            return None, None

    def run(self):
        """Executa a aplicação Streamlit."""
        st.title("Análise de Correlação e ANOVA")

        # Upload do arquivo CSV
        file_path = st.file_uploader("Carregue seu arquivo CSV", type=['csv'])

        if file_path is not None:
            if self.load_data(file_path):
                self.display_data()
                self.plot_correlation_matrix()

                # Análise ANOVA
                st.header("Análise de Variáveis Categóricas")
                categorical_columns = self.data.select_dtypes(include=['object']).columns
                numerical_columns = self.data.select_dtypes(include=['number']).columns

                if not categorical_columns.any() or not numerical_columns.any():
                    st.warning("Os dados não contêm colunas categóricas ou numéricas suficientes.")
                    return

                categorical_var = st.selectbox("Selecione a variável categórica:", categorical_columns)
                numerical_var = st.selectbox("Selecione a variável numérica:", numerical_columns)

                if st.button("Realizar ANOVA"):
                    f_statistic, p_value = self.perform_anova(categorical_var, numerical_var)
                    if f_statistic is not None and p_value is not None:
                        st.write(f"**Estatística F:** {f_statistic:.2f}")
                        st.write(f"**Valor p:** {p_value:.4f}")

                        # Interpretação dos resultados
                        if p_value < 0.05:
                            st.success("Rejeitamos a hipótese nula: há uma diferença significativa entre os grupos.")
                        else:
                            st.info("Não rejeitamos a hipótese nula: não há diferença significativa entre os grupos.")

# Executa o aplicativo
if __name__ == "__main__":
    app = DataAnalysisApp()
    app.run()

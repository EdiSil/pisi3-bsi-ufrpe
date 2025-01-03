import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load data
file_path = 'Datas/1_Cars_dataset_processado.csv'
cars_data = pd.read_csv(file_path)

# Set up Streamlit dashboard
st.set_page_config(page_title="Car Dataset Analysis", layout="wide")
st.title("Car Dataset Interactive Dashboard")

# Sidebar for filters
st.sidebar.header("Filters")
selected_brands = st.sidebar.multiselect(
    "Select Brands:", options=cars_data["marca"].unique(), default=cars_data["marca"].unique()
)
selected_fuel = st.sidebar.multiselect(
    "Select Fuel Types:", options=cars_data["combustivel"].unique(), default=cars_data["combustivel"].unique()
)

# Filter data based on selection
filtered_data = cars_data[
    (cars_data["marca"].isin(selected_brands)) &
    (cars_data["combustivel"].isin(selected_fuel))
]

# Set consistent color palette
palette = sns.color_palette("tab10", n_colors=filtered_data["marca"].nunique())

# Create plots
# 1. Count plot of car brands
st.subheader("Number of Cars by Brand")
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(data=filtered_data, x="marca", palette=palette, ax=ax)
ax.set_title("Number of Cars by Brand")
ax.set_xlabel("Brand")
ax.set_ylabel("Count")
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# 2. Box plot of price by car brand
st.subheader("Price Distribution by Brand")
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=filtered_data, x="marca", y="preco", palette=palette, ax=ax)
ax.set_title("Price Distribution by Brand")
ax.set_xlabel("Brand")
ax.set_ylabel("Price")
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# 3. Scatter plot of price vs. mileage
st.subheader("Price vs. Mileage by Brand")
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=filtered_data, x="quilometragem", y="preco", hue="marca", palette=palette, ax=ax)
ax.set_title("Price vs. Mileage by Brand")
ax.set_xlabel("Mileage (km)")
ax.set_ylabel("Price")
st.pyplot(fig)

# 4. Distribution plot of car prices
st.subheader("Distribution of Car Prices")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=filtered_data, x="preco", hue="marca", palette=palette, kde=True, ax=ax)
ax.set_title("Distribution of Car Prices")
ax.set_xlabel("Price")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# 5. Heatmap of correlation between numerical features
st.subheader("Correlation Heatmap")
corr = filtered_data.select_dtypes(include=["float64", "int64"]).corr()
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)

# 6. Count plot of cars by fuel type and transmission
st.subheader("Fuel Type by Transmission")
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(data=filtered_data, x="combustivel", hue="transmissão", ax=ax)
ax.set_title("Fuel Type by Transmission")
ax.set_xlabel("Fuel Type")
ax.set_ylabel("Count")
st.pyplot(fig)

# 7. Violin plot of mileage by car type
st.subheader("Mileage Distribution by Car Type")
fig, ax = plt.subplots(figsize=(10, 5))
sns.violinplot(data=filtered_data, x="tipo", y="quilometragem", palette="muted", ax=ax)
ax.set_title("Mileage Distribution by Car Type")
ax.set_xlabel("Car Type")
ax.set_ylabel("Mileage (km)")
st.pyplot(fig)

# 8. Line plot of average price over year range
st.subheader("Average Price by Year Range")
avg_price_year = filtered_data.groupby("year_range")["preco"].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=avg_price_year, x="year_range", y="preco", marker="o", ax=ax)
ax.set_title("Average Price by Year Range")
ax.set_xlabel("Year Range")
ax.set_ylabel("Average Price")
st.pyplot(fig)

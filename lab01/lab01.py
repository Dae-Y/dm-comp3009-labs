# Data Mining Lab 01
# Environment Preparation and Python Refresher
# Daehwan Yeo

import pandas as pd                 # for data manipulation and analysis
import matplotlib.pyplot as plt     # for basic charts
import seaborn as sns               # for cleaner and more informative plots
import io

# Section 2:

# Load the dataset into a DataFrame
df = pd.read_csv("bank.csv")

print("\n--- Section 2 Check and Explore the Dataset ---")

# View the First 5 Rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Get the Shape of the Dataset (rows, columns)
print(f"\nShape of the dataset: {df.shape}")

# Check the Column Names and Data Types
print("\nColumn names and data types:")
print(df.info())

# Check for Missing Values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Summary Statistics (for numerical columns)
print("\nSummary statistics for numerical columns:")
print(df.describe())



print("\n--- Section 2 Column Access and Filtering ---")

# Access a single column (returns a Pandas Series)
print("\n'income' column (as a Series):")
income_series = df['income']
print(income_series.head())

# Access multiple columns (returns a new DataFrame)
print("\n'age', 'sex', and 'income' columns (as a DataFrame):")
subset_df = df[['age', 'sex', 'income']]
print(subset_df.head())

# Filtering Rows with Conditions

# Clients older than 50
print("\nClients older than 50:")
older_clients = df[df['age'] > 50]
print(older_clients)

# Clients from the 'TOWN' region
print("\nClients from the 'TOWN' region:")
town_clients = df[df['region'] == 'TOWN']
print(town_clients)

# Clients with income greater than 30,000 AND who are married
print("\nClients with income > 30,000 and are married:")
high_income_married = df[(df['income'] > 30000) & (df['married'] == 'YES')]
print(high_income_married)

# Clients who are either from INNER_CITY OR have a mortgage
print("\nClients from INNER_CITY or have a mortgage:")
inner_city_or_mortgage = df[(df['region'] == 'INNER_CITY') | (df['mortgage'] == 'YES')]
print(inner_city_or_mortgage)


print("\n--- Section 3: Visualizing the Dataset ---")
print("Generating plots... Check for saved image files.")

# Set a consistent style for the plots
sns.set_style("whitegrid")

# --- Plot 1: Histogram of Age Distribution ---
# Histogram of Age Distribution bins=10

plt.figure(figsize=(8, 6)) # Create a new figure
plt.hist(df['age'], bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('age_distribution.png', dpi=300)
print("Saved 'age_distribution.png'")
plt.close() # Close the figure to free memory


# --- Plot 2: Bar Chart of PEP Subscription ---
plt.figure(figsize=(8, 6))
df['pep'].value_counts().plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Personal Equity Plan (PEP) Subscription')
plt.xlabel('PEP Subscription (Yes/No)')
plt.ylabel('Number of Clients')
plt.xticks(rotation=0) # Keep labels horizontal
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('PEP_subscription.png', dpi=300)
print("Saved 'PEP_subscription.png'")
plt.close()


# --- Plot 3: Box Plot of Income by Region ---
plt.figure(figsize=(8, 6))
sns.boxplot(x='region', y='income', data=df)
plt.title('Income Distribution by Region')
plt.xlabel('Region')
plt.ylabel('Income')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('income_by_region.png', dpi=300)
print("Saved 'income_by_region.png'")
plt.close()

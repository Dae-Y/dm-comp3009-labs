# Data Mining Lab 02
# Data Preparation
# Daehwan Yeo

# Load the modules that we'll be using
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# Load the dataset into a DataFrame
df = pd.read_csv("titanic.csv")


print("--- 1. Displaying Dataset Rows ---")
# Display the first 10 rows of the dataset.
print("\nFirst 10 rows:")
print(df.head(10))

# Display the first 5 rows of the dataset.
print("\nFirst 5 rows (default for head):")
print(df.head(5))

# Display the last 5 rows of the dataset.
print("\nLast 5 rows:")
print(df.tail())


print("\n\n--- 2. Examining Data Types ---")
# Examine the column names and data types as interpreted by pandas.
print("\nData types as loaded by pandas:")
print(df.info())
print("\nComments on Data Types:")
print("- Most types are appropriate (int64, float64, object for strings).")
print("- 'Survived' and 'Pclass' are numerical but represent categories, so they could be changed to 'category' type for more specific analysis.")
print("- 'PassengerId' is an identifier and not a value for calculation, so its integer type is fine but should not be used in mathematical summaries.")


print("\n\n--- 3. Data Dimensionality ---")
# Explain that the data is multidimensional because it's a table with rows and columns.
print("\nThe data is multidimensional, organized in a 2D table structure.")

# Find the number of attributes (columns) and the number of instances (rows).
num_instances, num_attributes = df.shape
print(f"The dataset has {num_instances} instances (rows).")
print(f"The dataset has {num_attributes} attributes (columns).")


print("\n\n--- 4. Summarising the Data ---")
# Check for any missing entries for each attribute.
print("\nMissing values per column:")
print(df.isnull().sum())

# Find the min, max, avg, and sum values for numeric columns.
print("\nSummary statistics for numeric columns:")
print(df.describe())


# if we run a funciton on the data frame it runs on ech column.
# `numeric_only=True` means that we ignore the text columns
# Calculate the sum for each numeric column.
print("\nSum of each numeric column:")
print(df.sum(numeric_only=True))


# Plot a histogram of all the numeric data together.
print("\nGenerating and saving histograms...")
df.hist(figsize=(12, 10), bins=20)
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.savefig('titanic_all_histograms.png')
print("Saved 'titanic_all_histograms.png'")
plt.close() # Close the plot to free up memory


# Plot a histogram for just the 'Age' column.
df['Age'].hist()
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('titanic_age_histogram.png')
print("Saved 'titanic_age_histogram.png'")
plt.close()



print("\n\n--- 5. Selecting Passengers with 'Dr.' in Their Name ---")
# Filter rows where the passenger's name contains the title 'Dr.'.
# The backslash before the dot escapes the special meaning of '.' in regex.
dr_filter = df['Name'].str.contains('Dr\.', na=False)

# Apply the filter to get a new DataFrame of only passengers with 'Dr.' in their name.
doctors_df = df[dr_filter]
print("\nPassengers with the title 'Dr.':")
print(doctors_df[['Name', 'Sex', 'Age']])

# Group the doctors by 'Sex' and count the number of passengers in each group.
doctor_counts_by_sex = doctors_df.groupby('Sex')['Name'].count()

# Display the result of the grouping.
print("\nNumber of doctors grouped by sex:")
print(doctor_counts_by_sex)



print("\n\n--- 6. Computing Average Fare by Class and Embarkation Port ---")
# Group by passenger class and embarkation port, then calculate the mean fare.
avg_fare_by_class_port = df.groupby(['Pclass', 'Embarked'])['Fare'].mean()

# Display the resulting summary.
print("\nAverage fare grouped by Pclass and Embarked:")
print(avg_fare_by_class_port)
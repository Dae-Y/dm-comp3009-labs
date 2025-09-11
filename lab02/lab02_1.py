'''
Data Mining Lab02 - Data Preparation
Daehwan Yeo

'''

# Load the modules that we'll be using
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# Load the dataset into a DataFrame
df = pd.read_csv("titanic.csv")

# Open a results file to write all textual outputs
with open("ex1_result.txt", "w", encoding="utf-8") as out:

    out.write("--- 1. Displaying Dataset Rows ---\n")
    # Display the first 10 rows of the dataset.
    out.write("\nFirst 10 rows:\n")
    out.write(df.head(10).to_string(index=True))
    out.write("\n")

    # Display the first 5 rows of the dataset.
    out.write("\nFirst 5 rows (default for head):\n")
    out.write(df.head(5).to_string(index=True))
    out.write("\n")

    # Display the last 5 rows of the dataset.
    out.write("\nLast 5 rows:\n")
    out.write(df.tail().to_string(index=True))
    out.write("\n")

    out.write("\n\n--- 2. Examining Data Types ---\n")
    # Examine the column names and data types as interpreted by pandas.
    out.write("\nData types as loaded by pandas:\n")
    info_buf = io.StringIO()
    df.info(buf=info_buf)
    out.write(info_buf.getvalue())

    out.write("\nComments on Data Types:\n")
    out.write("- Most types are appropriate (int64, float64, object for strings).\n")
    out.write("- 'Survived' and 'Pclass' are numerical but represent categories, so they could be changed to 'category' type for more specific analysis.\n")
    out.write("- 'PassengerId' is an identifier and not a value for calculation, so its integer type is fine but should not be used in mathematical summaries.\n")

    out.write("\n\n--- 3. Data Dimensionality ---\n")
    # Explain that the data is multidimensional because it's a table with rows and columns.
    out.write("\nThe data is multidimensional, organized in a 2D table structure.\n")

    # Find the number of attributes (columns) and the number of instances (rows).
    num_instances, num_attributes = df.shape
    out.write(f"The dataset has {num_instances} instances (rows).\n")
    out.write(f"The dataset has {num_attributes} attributes (columns).\n")

    out.write("\n\n--- 4. Summarising the Data ---\n")
    # Check for any missing entries for each attribute.
    out.write("\nMissing values per column:\n")
    out.write(df.isnull().sum().to_string())
    out.write("\n")

    # Find the min, max, avg, and sum values for numeric columns.
    out.write("\nSummary statistics for numeric columns:\n")
    out.write(df.describe().to_string())
    out.write("\n")

    # Calculate the sum for each numeric column.
    out.write("\nSum of each numeric column:\n")
    out.write(df.sum(numeric_only=True).to_string())
    out.write("\n")

    # Plot a histogram of all the numeric data together.
    out.write("\nGenerating and saving histograms...\n")
    df.hist(figsize=(12, 10), bins=20)
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.savefig('titanic_all_histograms.png')
    plt.close()
    out.write("Saved 'titanic_all_histograms.png'\n")

    # Plot a histogram for just the 'Age' column.
    df['Age'].hist()
    plt.title('Age Distribution of Passengers')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig('titanic_age_histogram.png')
    plt.close()
    out.write("Saved 'titanic_age_histogram.png'\n")

    out.write("\n\n--- 5. Selecting Passengers with 'Dr.' in Their Name ---\n")
    # Filter rows where the passenger's name contains the title 'Dr.'.
    # The backslash before the dot escapes the special meaning of '.' in regex.
    dr_filter = df['Name'].str.contains(r'Dr\.', na=False)

    # Apply the filter to get a new DataFrame of only passengers with 'Dr.' in their name.
    doctors_df = df[dr_filter]
    out.write("\nPassengers with the title 'Dr.':\n")
    if not doctors_df.empty:
        out.write(doctors_df[['Name', 'Sex', 'Age']].to_string(index=False))
    else:
        out.write("None found.")
    out.write("\n")

    # Group the doctors by 'Sex' and count the number of passengers in each group.
    doctor_counts_by_sex = doctors_df.groupby('Sex')['Name'].count()

    # Display the result of the grouping.
    out.write("\nNumber of doctors grouped by sex:\n")
    if not doctor_counts_by_sex.empty:
        out.write(doctor_counts_by_sex.to_string())
    else:
        out.write("No doctors to group.")
    out.write("\n")

    out.write("\n\n--- 6. Computing Average Fare by Class and Embarkation Port ---\n")
    # Group by passenger class and embarkation port, then calculate the mean fare.
    avg_fare_by_class_port = df.groupby(['Pclass', 'Embarked'])['Fare'].mean()

    # Display the resulting summary.
    out.write("\nAverage fare grouped by Pclass and Embarked:\n")
    out.write(avg_fare_by_class_port.to_string())
    out.write("\n")

# After the with-block, result.txt is fully written.

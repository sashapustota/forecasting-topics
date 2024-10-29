import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import seaborn as sns
import statsmodels.api as sm

# Set your directory path where the CSV files are stored
directory_path = 'data/categories/'
results_path = 'results/'

# Create the results folder if it doesn't exist
os.makedirs(results_path, exist_ok=True)

# Load all CSV files
csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

# Set Seaborn style for prettier plots
sns.set(style="whitegrid")

# Function to add stars based on p-value
def p_value_stars(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""

# Loop over each CSV file
for file in csv_files:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file)

    # Scale the topic values by 100 to convert proportions to percentages
    df.iloc[:, 1:] = df.iloc[:, 1:] * 100  # Assuming the first column is 'year'

    # Extract the category name from the filename (remove 'df_topics_' and '.csv')
    category_name = os.path.basename(file).replace('df_topics_', '').replace('.csv', '')

    # Create a folder for each CSV file in the results directory
    category_path = os.path.join(results_path, category_name)
    os.makedirs(category_path, exist_ok=True)

    # Create a color palette for the topics (ensure unique colors for each topic)
    palette = sns.color_palette("tab10", len(df.columns))

    # Loop over each column (topic) in the DataFrame
    for idx, column in enumerate(df.columns[1:]):
        # Prepare the data
        X = np.arange(len(df)).reshape(-1, 1)  # Feature matrix with time points
        y = df[column].values

        # Add quadratic term
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        # Fit the quadratic regression model
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)

        # Add a constant to X_poly for statsmodels
        X_poly_sm = sm.add_constant(X_poly)

        # Fit the model using statsmodels for p-values
        model_sm = sm.OLS(y, X_poly_sm).fit()
        p_values = model_sm.pvalues[1:]  # Exclude the constant term
        coefficients = model_sm.params[1:]  # Exclude the constant term

        # Extract the coefficients
        linear_coef, quadratic_coef = coefficients
        p_value_linear, p_value_quadratic = p_values

        # Determine stars for p-values
        stars_linear = p_value_stars(p_value_linear)
        stars_quadratic = p_value_stars(p_value_quadratic)
        print(stars_linear, stars_quadratic)
        print(category_name, column)

        # Plot the results
        plt.figure(figsize=(12, 6))  # Make the plot wider
        plt.scatter(X, y, color=palette[idx], label=f'{column} Data', marker='o')
        plt.plot(X, y_pred, color=palette[idx], linestyle='-', linewidth=2.5, label=f'Fit line')

        # Title with coefficients, p-values, and R2
        plt.title(f'{column}: Linear Coef = {linear_coef:.2f}{stars_linear}, '
                  f'Quadratic Coef = {quadratic_coef:.2f}{stars_quadratic}, R2 = {r2:.2f}')
        plt.xlabel('Year')
        plt.ylabel('Percentage')

        # Set the x-ticks to the correct 'year' values
        plt.xticks(ticks=X.flatten(), labels=df['year'], rotation=45)

        # Custom legend
        plt.legend(loc='best')

        plt.grid(True)

        # Save the plot
        output_filename = f"{column}_quadratic_regression.png"
        plt.savefig(os.path.join(category_path, output_filename))
        plt.close()

        #print(f"Saved plot for {column} in {category_name} with R2 {r2:.2f}")

print("All plots generated and saved.")
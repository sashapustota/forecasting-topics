import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import seaborn as sns

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

# Function to compute the confidence intervals
def compute_confidence_intervals(X, y_pred, model, alpha=0.05):
    # Number of samples
    n = len(X)
    
    # Predict errors
    y = model.predict(X)
    residuals = y_pred - y
    
    # Standard error
    s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
    
    # Confidence interval
    t_value = stats.t.ppf(1 - alpha/2, df=n-2)
    conf_interval = t_value * s_err * np.sqrt(1/n + (X - np.mean(X))**2 / np.sum((X - np.mean(X))**2))
    
    return y_pred - conf_interval, y_pred + conf_interval

# Loop over each CSV file
for file in csv_files:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file)

    # # Convert 'release_date' to datetime
    # df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # # Handle missing dates
    # if df['release_date'].isnull().any():
    #     print(f"Warning: Some 'release_date' entries are missing in file {file}")

    # # Convert 'release_date' to numerical values (days since the earliest date)
    # df['days_since_start'] = (df['release_date'] - df['release_date'].min()).dt.days

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

        # Remove all datapoints that are 0 from the column
        # mask = df[column] > 0
        # df_filtered = df[mask]

        # Prepare the data
        X = np.arange(len(df)).reshape(-1, 1)  # Assuming each row is a time point (e.g., a year)
        y = df[column].values

        # Fit the linear regression model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Calculate the slope and p-value
        slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
        stars = p_value_stars(p_value)
        r2 = model.score(X, y)

        # Plot the results
        plt.figure(figsize=(12, 6))  # Make the plot wider
        plt.scatter(X, y, color=palette[idx], label=f'{column} Data', marker='o')
        plt.plot(X, y_pred, color=palette[idx], linestyle='-', linewidth=2.5, label=f'Fit line')

        # Title with slope and p-value
        #plt.title(f'{column}: Slope = {slope:.2f}, p = {p_value:.3f}{stars}, R2 = {r2:.2f}')
        plt.title(f'{column}: Slope = {slope:.2f}{stars}, R2 = {r2:.2f}')
        plt.xlabel('Time')
        plt.ylabel('Percentage')

        #Set the x-ticks to the correct 'year' values
        plt.xticks(ticks=X.flatten(), labels=df['year'], rotation=45)

        # Custom legend without slope/p-value details
        plt.legend([f'{column} Data', f'Fit line'], loc='best')

        plt.grid(True)

        # Save the plot
        output_filename = f"{column}_regression.png"
        plt.savefig(os.path.join(category_path, output_filename))
        plt.close()

        #print(f"Saved plot for {column} in {category_name} with slope {slope:.2f} and p-value {p_value:.3f}")

print("All plots generated and saved.")
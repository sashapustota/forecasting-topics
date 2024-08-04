import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the DataFrame with topics
df = pd.read_csv('data/df_topics.csv')

# Load the d values from the p_and_d_determination.py script
d = {}
with open('data/d_values.txt', 'r') as f:
    for line in f:
        topic, value = line.strip().split(':')
        d[topic] = int(value)

# Create an empty DataFrame to store the results
results = pd.DataFrame(columns=['RMSE', 'Naive RMSE', 'MAE', 'Naive MAE'])

# Fit ARIMA models, generate forecasts, and evaluate
for topic in df.columns:
    train = df[topic][:-5]
    test = df[topic][-5:]
    
    # Fit ARIMA model
    model = auto_arima(train, d=d[topic], start_p=0, start_q=0, seasonal=False)
    
    # Save model summary
    with open(f'models/model_summary_{topic}.txt', 'w') as f:
        f.write(model.summary().as_text())
    
    # Forecast and confidence intervals
    forecast, conf_int = model.predict(n_periods=5, return_conf_int=True)
    conf_int = pd.DataFrame(conf_int, index=test.index, columns=['lower', 'upper'])
    forecast.index = test.index

    # Mean naive forecast
    mean_naive_forecast = pd.Series(np.full(len(test), train.mean()), index=test.index)

    # Add the last year of the training set to the forecast and the test set, for plots
    forecast = pd.concat([train[-1:], forecast])
    test = pd.concat([train[-1:], test])
    mean_naive_forecast = pd.concat([train[-1:], mean_naive_forecast])
    
    # Calculate RMSE and MAE
    rmse = round(np.sqrt(mean_squared_error(test, forecast)), 3)
    mae = round(mean_absolute_error(test, forecast), 3)
    rmse_naive = round(np.sqrt(mean_squared_error(test, mean_naive_forecast)), 3)
    mae_naive = round(mean_absolute_error(test, mean_naive_forecast), 3)

    # Store results
    results.loc[topic] = [rmse, rmse_naive, mae, mae_naive]

    # Plot forecasts
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(forecast, label='Forecast')
    plt.plot(mean_naive_forecast, label='Mean Naive Forecast')
    plt.fill_between(conf_int.index, conf_int['lower'], conf_int['upper'], color='k', alpha=0.07)
    plt.legend(loc='lower left')
    plt.title(f'Forecast for {topic}')
    plt.savefig(f'plots/forecast_{topic}.png')
    plt.close()

# Save the results as a CSV file
results.to_csv('results/results.csv')
print("Results saved to 'results/results.csv'.")

# Plot all forecasts in a single figure
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
for i, topic in enumerate(df.columns):
    train = df[topic][:-5]
    test = df[topic][-5:]
    model = auto_arima(train, d=d[topic], start_p=0, start_q=0, seasonal=False)
    forecast, conf_int = model.predict(n_periods=5, return_conf_int=True)
    conf_int = pd.DataFrame(conf_int, index=test.index, columns=['lower', 'upper'])
    forecast.index = test.index
    mean_naive_forecast = pd.Series(np.full(len(test), train.mean()), index=test.index)
    # Add the last year of the training set to the forecast and the test set, for plots
    forecast = pd.concat([train[-1:], forecast])
    test = pd.concat([train[-1:], test])
    mean_naive_forecast = pd.concat([train[-1:], mean_naive_forecast])
    axs[i//2, i%2].plot(train, label='Train')
    axs[i//2, i%2].plot(test, label='Test')
    axs[i//2, i%2].plot(forecast, label='Forecast')
    axs[i//2, i%2].plot(mean_naive_forecast, label='Mean Naive Forecast')
    axs[i//2, i%2].fill_between(conf_int.index, conf_int['lower'], conf_int['upper'], color='k', alpha=0.07)
    axs[i//2, i%2].legend(loc='lower left')
    axs[i//2, i%2].set_title(f'Forecast for {topic}')
plt.tight_layout()
plt.savefig('plots/all_forecasts.png')
plt.close()

# Print a confirmation message
print("All forecasts saved to 'plots/'.")
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from pmdarima.arima.utils import ndiffs

# Load the DataFrame with topics
df = pd.read_csv('data/df_topics.csv')

# Plot the PACF plots for each topic
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
for i, topic in enumerate(df.columns):
    plot_pacf(df[topic], ax=axs[i//2, i%2], title=f'PACF for {topic}')
plt.tight_layout()
plt.savefig('plots/pacf_all_topics.png')
plt.close()

# Print a confirmation message
print("PACF plots saved to 'plots/pacf_all_topics.png'.")

# Determine the optimal d parameter for each topic
d = {}
for topic in df.columns:
    d[topic] = ndiffs(df[topic], test='kpss')
    print(f'Number of differences needed for {topic}: {d[topic]}')

# Save the d values to a file for use in the next script
with open('data/d_values.txt', 'w') as f:
    for topic, value in d.items():
        f.write(f'{topic}:{value}\n')

# Print a confirmation message
print("d values saved to 'data/d_values.txt'.")

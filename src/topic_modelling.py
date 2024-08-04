import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction

# Set a random seed
random.seed(420)
# Pick a random number to be used for the random_state of the clustering algorithm
random_number = random.randint(0, 1000000)
print(f"Random number for clustering: {random_number}")

# Load the cleaned DataFrame
df = pd.read_csv('data/df_clean.csv')

# Initialize KMeans clustering model
cluster_model = KMeans(n_clusters=13, random_state=random_number)
# Initialize an empty dimensionality reduction model
empty_dimensionality_model = BaseDimensionalityReduction()

# Initialize and fit BERTopic model
topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",
    umap_model=empty_dimensionality_model,
    hdbscan_model=cluster_model
)
topics, probs = topic_model.fit_transform(df['text_clean'])

# Generate topic labels
topic_labels = topic_model.generate_topic_labels(
    nr_words=3,
    topic_prefix=False,
    word_length=15,
    separator=" - "
)
topic_model.set_topic_labels(topic_labels)
print("Here are the topics for your data:")
print(topic_model.get_topic_info())

# Add the topics for each 'text_clean' to the dataframe
df['Topic'] = topics

# Create a DataFrame from topic_model.get_topic_info()
topic_info = topic_model.get_topic_info()

# Create a mapping dictionary for Topic to CustomName
topic_mapping = dict(zip(topic_info['Topic'], topic_info['CustomName']))

# Map the topics from df to the CustomName
df['Topic'] = df['Topic'].map(topic_mapping)

# Aggregate data by year and drop topics that are not of interest
total_paragraphs_per_year = df.groupby('year').size()
paragraphs_per_topic_per_year = df.groupby(['year', 'Topic']).size().unstack(fill_value=0)
proportion_per_topic_per_year = paragraphs_per_topic_per_year.div(total_paragraphs_per_year, axis=0)

# Filter the proportion_per_topic_per_year DataFrame to include only the topics of interest
topics_of_interest = [
    'physic - information - system', 
    'qubits - qubit - computer', 
    'sensor - measurement - use', 
    'atom - ion - trap', 
    'communication - security - cryptography', 
    'photon - optical - source'
]

proportion_per_topic_per_year = proportion_per_topic_per_year[topics_of_interest]

# Rename the topics to formal names
proportion_per_topic_per_year.columns = [
    'Foundational Research', 
    'Computing Components', 
    'Sensing and Measurement', 
    'Systems and Manipulation Techniques', 
    'Cryptography and Secure Communication', 
    'Photonics'
]

# Plot the normalized data
proportion_per_topic_per_year.plot(kind='line', marker='o')
plt.gcf().set_size_inches(12, 6)
plt.title('Proportion of Topics in Quantum National Strategies by Year')
plt.xlabel('Year')
plt.ylabel('Proportion of Paragraphs')
plt.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('plots/proportion_per_topic_per_year.png')
plt.close()

# Save the DataFrame with topics to a CSV file
proportion_per_topic_per_year.to_csv('data/df_topics.csv', index=False)
print("Data with topics saved to 'data/df_topics.csv'.")
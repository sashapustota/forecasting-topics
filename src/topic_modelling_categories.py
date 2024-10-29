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

# Load the namings of the topics
df_topics = pd.read_csv('data/topic_names.csv')

# Put unique 'Category' values in the df_topics DataFrame into a list
categories = df_topics['Category'].unique()

categories = ['International Collaboration']

for category in categories:

    # Load the cleaned DataFrame
    df = pd.read_csv('data/df_clean.csv')

    # Load the namings of the topics
    df_topics = pd.read_csv('data/topic_names.csv')

    # Print "processing category" message
    print(f"Processing category: {category}")

    # Keep only the "National Strategy and Recommendations" Category
    df = df[df['Category'] == category]

    # Reset the index of the DataFrame
    df.reset_index(drop=True, inplace=True)

    # Filter df_topics by category
    df_topics = df_topics[df_topics['Category'] == category]

    # Get the "n_of_topics" value for category in the df_topics DataFrame
    n_of_topics = df_topics[df_topics['Category'] == category]['n_of_topics'].values[0]

    # Print the number of topics
    print(f"Number of topics for {category} Category: {n_of_topics}")

    # Initialize KMeans clustering model
    cluster_model = KMeans(n_clusters=n_of_topics, random_state=random_number)
    # Initialize an empty dimensionality reduction model
    empty_dimensionality_model = BaseDimensionalityReduction()

    # Initialize and fit BERTopic model
    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        umap_model=empty_dimensionality_model,
        hdbscan_model=cluster_model
    )
    topics, probs = topic_model.fit_transform(df['text_clean'])

    if category == "International Collaboration":

        topics_to_merge = [[0,3]]
        topic_model.merge_topics(df['text_clean'], topics_to_merge)

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

    # Use the topic names from the df_topics DataFrame (column 'original_name') to rename the topics (column 'new_name') in the proportion_per_topic_per_year DataFrame "Topic" column
    df['Topic'] = df['Topic'].map(df_topics.set_index('original_name')['new_name'])

    # Aggregate data by year and drop topics that are not of interest
    total_paragraphs_per_year = df.groupby('year').size()
    paragraphs_per_topic_per_year = df.groupby(['year', 'Topic']).size().unstack(fill_value=0)
    proportion_per_topic_per_year = paragraphs_per_topic_per_year.div(total_paragraphs_per_year, axis=0)

    # Only include years 2014 and later
    # proportion_per_topic_per_year = proportion_per_topic_per_year.loc[2014:]

    # Reset the index to ensure 'year' is a column, not an index
    proportion_per_topic_per_year.reset_index(inplace=True)

    ### 
    # Aggregate data by year, topic, and pdf_name (this is for manual inspection)
    total_paragraphs_per_year_topic = df.groupby(['year', 'Topic']).size()
    paragraphs_per_pdf_per_year_topic = df.groupby(['year', 'Topic', 'pdf_name']).size().unstack(fill_value=0)

    # Calculate proportion of each pdf's contribution to a topic in a given year
    proportion_per_pdf_per_topic_per_year = paragraphs_per_pdf_per_year_topic.div(total_paragraphs_per_year_topic, axis=0)

    # Reset the index to ensure 'year' and 'Topic' are columns, not indices
    proportion_per_pdf_per_topic_per_year.reset_index(inplace=True)

    # Save this complementary DataFrame to a CSV file
    proportion_per_pdf_per_topic_per_year.to_csv(f'data/categories/pdf_contributions_{category}.csv', index=False)
    print(f"PDF contributions data saved to 'data/categories/pdf_contributions_{category}.csv'.")

    # Get nunique() of the 'Topic' column
    unique_topics = df['Topic'].nunique()

    # Use tab20 colormap
    cmap = plt.get_cmap('tab20', unique_topics)

    # Plot the normalized data with year as the x-axis and all other columns as y
    proportion_per_topic_per_year.plot(
        x='year',  # Specify 'year' as the x-axis
        y=proportion_per_topic_per_year.columns.drop('year'),  # All other columns except 'year' are y-values
        kind='line', 
        marker='o', 
        color=cmap.colors
    )
    plt.gcf().set_size_inches(14, 6)
    plt.title(f'Proportion of Topics in {category} Category by Year')
    plt.xlabel('Year')
    plt.ylabel('Proportion of Paragraphs')
    plt.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/categories/proportion_per_topic_per_year_{category}.png')
    # Print confirmation message
    print(f"Plot saved to 'plots/categories/proportion_per_topic_per_year_{category}.png'.")
    plt.close()

    # Save the DataFrame with topics to a CSV file
    proportion_per_topic_per_year.to_csv(f'data/categories/df_topics_{category}.csv', index=False)
    df.to_csv(f'data/categories/full_df_topics_{category}.csv', index = False)
    print(f"Data with topics saved to 'data/categories/df_topics_{category}.csv'.")
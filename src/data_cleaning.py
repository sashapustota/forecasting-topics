import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def nltk2wn_tag(nltk_tag):
    """Converts NLTK POS tags to WordNet POS tags."""
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence):
    """Lemmatizes a sentence using POS tagging."""
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))    
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(res_words)

def clean_data(df, column_name, filtered_keywords):
    """Cleans the text data in the DataFrame."""
    # Making a new column titled "text_clean" which is a copy of the column for processing and making it lowercase.
    df['text_clean'] = df[column_name].str.lower()
    
    # Removing punctuation and special characters
    df['text_clean'] = df['text_clean'].str.replace('[^\w\s]', '', regex=True)
    df['text_clean'] = df['text_clean'].str.replace('[^A-Za-z0-9]+', ' ', regex=True)
    
    # Remove stopwords
    stop_words = stopwords.words('english')
    df['text_clean'] = df['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words and word not in filtered_keywords]))
    
    # Lemmatize the text
    df['text_clean'] = df['text_clean'].apply(lambda x: lemmatize_sentence(x))
    
    return df

def main():
    # Load the main DataFrame
    df = pd.read_csv('data/df_filtered.csv')
    print(f"Rows before removal: {len(df)}")
    # Stop the program
    # return

    # Load the DataFrame containing rows to remove
    df_temporal_removal = pd.read_csv('data/manual_fix.csv')
    
    # Remove the specified rows from the main DataFrame
    df = df[~df.set_index(['page', 'para_num', 'pdf_name']).index.isin(df_temporal_removal.set_index(['page', 'para_num', 'pdf_name']).index)]
    print(f"Rows after removal: {len(df)}")
    
    # Remove NAs
    df.dropna(subset=['text'], inplace=True)
    
    # Remove all text that is shorter than 150 chars (50th percentile) but keep all text that is a bullet point
    df = df[(df['text'].str.len() >= df['text'].apply(len).quantile(0.50)) | (df['text'].str.contains('^•', na=False))]
    
    # Remove all 'text' entries that mention websites
    df = df[~df['text'].str.contains(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', na=False)]
    
    # Create a keyword list
    filtered_keywords = df['country'].unique().tolist()
    filtered_keywords += [country.lower() for country in ['quantum','th', 'de', 'al', 'ce', 'rev', '115', 'page', 'australias', 'australian', 'denmark', 'danish', 'europe', 'eu', 'european', 'scotland', 'korea', 'ireland']]
    filtered_keywords = list(set(filtered_keywords))
    
    # Clean the data
    df = clean_data(df, 'text', filtered_keywords)
    
    # Remove 'Miscellaneous' category from the DataFrame
    df = df[df['Category'] != 'Miscellaneous']
    
    # Re-index the DataFrame
    df.reset_index(drop=True, inplace=True)
    
    # Save the cleaned DataFrame to a CSV file
    df.to_csv('data/df_clean.csv', index=False)
    print(f"Rows after removal: {len(df)}")
    print("Data cleaning completed. Cleaned data saved to 'df_clean.csv'.")

if __name__ == "__main__":
    main()

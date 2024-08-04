import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def load_dataframe(csv_path):
    return pd.read_csv(csv_path)

def extract_text_from_pdf(pdf_name):
    pdf_texts = []
    pdf_file_path = "documents/" + pdf_name + ".pdf"

    with open(pdf_file_path, 'rb') as pdfFileObj:
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        for page_num in range(pdfReader.numPages):
            page = pdfReader.getPage(page_num)
            pdf_texts.append(page.extractText())
    
    return pdf_texts

def calculate_cosine_similarity(gpt_text, pdf_text):
    if not gpt_text.strip() or not pdf_text.strip():
        return None

    vectorizer = TfidfVectorizer()
    corpus = [gpt_text, pdf_text]
    X = vectorizer.fit_transform(corpus)
    cos_sim = cosine_similarity(X[0], X[1])[0][0]
    
    return cos_sim

def process_pdf_pages(df, pdf_name):
    pdf_texts = extract_text_from_pdf(pdf_name)
    df_paragraphs = pd.DataFrame(columns=['pdf_name', 'page', 'gpt_text', 'pdf_text', 'cosine_similarity', 'text_length_diff'])
    # Convert all text values to string
    df['text'] = df['text'].astype(str)

    for page_num, pdf_text in enumerate(pdf_texts, start=1):
        gpt_text = ' '.join(df[(df['pdf_name'] == pdf_name) & (df['page'] == page_num)]['text'].values)
        text_length_diff = abs(len(gpt_text) - len(pdf_text))
        cos_sim = calculate_cosine_similarity(gpt_text, pdf_text)
        
        df_paragraphs = df_paragraphs.append({
            'pdf_name': pdf_name,
            'page': page_num,
            'gpt_text': gpt_text,
            'pdf_text': pdf_text,
            'cosine_similarity': cos_sim,
            'text_length_diff': text_length_diff
        }, ignore_index=True)

    df_paragraphs['cosine_similarity'] = df_paragraphs['cosine_similarity'].fillna(0)

    return df_paragraphs

def generate_cosine_similarity_report(df_paragraphs, threshold=0.80):

    pages_with_low_cosine_similarity = {}
    num_pages = {}
    total_pages = {}

    for pdf_name in df_paragraphs['pdf_name'].unique():
        pages_with_low_cosine_similarity[pdf_name] = df_paragraphs[(df_paragraphs['pdf_name'] == pdf_name) & (df_paragraphs['cosine_similarity'] < threshold)]['page'].tolist()
        num_pages[pdf_name] = len(pages_with_low_cosine_similarity[pdf_name])
        total_pages[pdf_name] = df_paragraphs[df_paragraphs['pdf_name'] == pdf_name]['page'].nunique()

    df_pages_with_low_cosine_similarity = pd.DataFrame(pages_with_low_cosine_similarity.items(), columns=['pdf_name', 'pages_with_low_cosine_similarity'])
    df_num_pages = pd.DataFrame(num_pages.items(), columns=['pdf_name', 'num_pages'])
    df_total_pages = pd.DataFrame(total_pages.items(), columns=['pdf_name', 'total_pages'])

    df_result = pd.merge(df_pages_with_low_cosine_similarity, df_num_pages, on='pdf_name')
    df_result = pd.merge(df_result, df_total_pages, on='pdf_name')
    df_result['percentage_pages'] = (df_result['num_pages'] / df_result['total_pages']) * 100

    return df_result

def plot_cosine_similarity_distribution(df_paragraphs):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_paragraphs['cosine_similarity'], fill=True)
    plt.xlim(0, 1)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Density Distribution of Cosine Similarity Values')
    plt.savefig('plots/cosine_similarity_distribution.png')
    plt.close()

def filter_low_similarity_pages(df, df_result):
    df_filtered = pd.DataFrame(columns=df.columns)

    for _, row in df_result.iterrows():
        pdf_name = row['pdf_name']
        pages_with_low_cosine_similarity = row['pages_with_low_cosine_similarity']
        
        df_filtered = pd.concat([df_filtered, df[(df['pdf_name'] == pdf_name) & (~df['page'].isin(pages_with_low_cosine_similarity))]], axis=0)

    return df_filtered

def main():
    csv_path = 'data/df_full.csv'
    df = load_dataframe(csv_path)

    all_df_paragraphs = pd.DataFrame()

    for pdf_name in df['pdf_name'].unique():
        df_paragraphs = process_pdf_pages(df, pdf_name)
        all_df_paragraphs = pd.concat([all_df_paragraphs, df_paragraphs])

    #plot_cosine_similarity_distribution(all_df_paragraphs)

    df_result = generate_cosine_similarity_report(all_df_paragraphs)
    print(f"Mean: {df_result['percentage_pages'].mean()}")
    print(f"Standard Deviation: {df_result['percentage_pages'].std()}")

    df_filtered = filter_low_similarity_pages(df, df_result)

    # Save the filtered DataFrame
    df_filtered.to_csv('data/df_filtered.csv', index=False)

    # Print the confirmation message
    print("Filtered DataFrame saved to 'data/df_filtered.csv'.")

if __name__ == "__main__":
    main()

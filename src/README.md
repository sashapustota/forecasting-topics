# Source Code for Quantum Strategy Analysis

This folder contains the source code necessary for the project - data processing, topic modeling, and fitting of the time series forecasting models to predict the evolution of popularity of quantum physics research areas.

## Folder Structure

- **src/**
  - `pdf_comparison.py`: Script for comparing the text extracted from PDFs using GPT3 with the original text.
  - `data_cleaning.py`: Script for cleaning and preprocessing the text data.
  - `topic_modeling.py`: Script for applying topic modeling using BERTopic.
  - `p_and_d_determination.py`: Script for determining the p and d values for ARIMA models.
  - `fit_arima.py`: Script for fitting ARIMA models and forecasting topic trends over time.

## Scripts Overview

### 1. `pdf_comparison.py`

**Purpose**:
This script compares the text extracted from PDFs using GPT3 with the original text to identify any discrepancies or hallucinations in the extraction process using cosine similarity. All of the pages that have a cosine similarity score below a certain threshold are removed from the dataset. The script saves a CSV file with the filtered data to the `data/` directory titled `data_filtered.csv`.

**Usage**:
```bash
python pdf_comparison.py
```

### 2. `data_cleaning.py`

**Purpose**:  
This script takes the previously created `data_filtered.csv` and performs several data cleaning procedures, including removing irrelevant text, filtering out short text entries, and removing stop words and special characters. It prepares the data for topic modeling by ensuring that only the most relevant and meaningful text is analyzed.

**Key Steps**:
- Loading data.
- Removing rows based on manual review (`manual_fix.csv`).
- Filtering out short text and text with URLs.
- Lemmatizing text and removing stopwords.
- Saving the cleaned data to `df_clean.csv`.

**Usage**:
```bash
python data_cleaning.py
```

### 3. `topic_modeling.py`

**Purpose**:
This script fits the BERTopic topic model to the cleaned data. 13 topics are identified in the text and each paragraph is assigned to one of the topics based on the content. The script saves the data with the assigned topics to `data/df_topics.csv`.

**Usage**:
```bash
python topic_modeling.py
```

### 4. `p_and_d_determination.py`

**Purpose**:
This script determines the ranges of p and d values for the ARIMA model.

**Key Steps**:
- Loading the data with topics.
- Plotting the PACF plots for each topic to determine the range of p values. Plots are saved to `plots/pacf_all_plots`.
- Determining the range of d values based on the differencing required to make the time series stationary. The dictionary with required *d* values is saved to `data/d_values.txt` and used in the next script.

**Usage**:
```bash
python p_and_d_determination.py
```

### 5. `fit_arima.py`

**Purpose**:
This script fits ARIMA models to the time series data for each topic and forecasts the topic popularit. The script saves model summaries for each topic to values to `models/model_summary_[topic].csv` and the plots of the forecasted values to `plots/forecast_[topic]`, as well as a combined plot of all forecasts to `plots/all_forecasts.png`.


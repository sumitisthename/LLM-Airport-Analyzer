# AeroInsights

AeroInsights is a Streamlit-based application designed to analyze and visualize airport reviews. It utilizes web scraping to gather reviews, performs sentiment analysis, and generates insights using machine learning models.

## Features

- **Scrape Reviews**: Extract reviews from specified URLs and save them for analysis.
- **Text Generation**: Generate text responses using a Hugging Face language model.
- **Sentiment Analysis**: Classify reviews into positive and negative sentiments.
- **Visualization**: Display word clouds and sentiment distribution graphs.

## Prerequisites

Ensure you have Python 3.7+ installed on your machine. You also need to install the required libraries listed in `requirements.txt`.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/sumitisthename/LLM-Airport-Analyzer.git
   cd LLM-Airport-Analyzer

2. **Clone the Repository:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt

## Usage

**Start the Streamlit Application**

```bash
streamlit run app.py

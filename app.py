import streamlit as st
from selenium import webdriver
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from transformers import pipeline
from wordcloud import WordCloud
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

'''
from streamlit_background import st_set_background
# Get the URL of the background image from the user
image_url = 'https://i.postimg.cc/0NjsRkSh/Airport.jpg'

# Set the background image using the custom component
st_set_background(image_url)
'''
# Load a model and tokenizer from Hugging Face
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate text using Hugging Face model
def generate_text(prompt, model, tokenizer, max_length=200):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Function to scrape reviews and save to 'reviews.txt'
def scrape(base_url, num_pages):
    if os.path.exists('reviews.txt'):
        os.remove('reviews.txt')
    driver = webdriver.Chrome()
    for page_num in range(1, num_pages + 1):
        url = f'{base_url}/page/{page_num}' if page_num > 1 else base_url
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        reviews = soup.find_all('div', class_='body')
        with open('reviews.txt', 'a', encoding='utf-8') as file:
            for i, review in enumerate(reviews):
                text = review.text.strip()
                file.write(f'Review {i + 1 + (page_num - 1) * len(reviews)}:\n{text}\n{"-" * 80}\n')
    driver.quit()

# Function to load text from 'reviews.txt'
def load_text(file_path='reviews.txt', encoding='utf-8'):
    try:
        with open(file_path, encoding=encoding) as f:
            text = f.read()
        return text
    except UnicodeDecodeError as e:
        st.error(f"Error decoding file {file_path} with encoding {encoding}: {e}")

# Function to perform text splitting
def split_text(data):
    # Split the content into individual reviews based on the delimiter "Review"
    docs = data.split("Review")[1:]
    docs = [review.strip() for review in docs]
    return docs

# Function to initialize Sentence Transformers embeddings
def initialize_sentence_transformers_embeddings(docs):
    gist_embedding = SentenceTransformer("avsolatorio/GIST-Embedding-v0")
    _list = gist_embedding.encode(docs, convert_to_tensor=True)
    return gist_embedding, _list

# Function to convert user question to embeddings using Faiss
def convert_question_to_embeddings(question, gist_embedding):
    search_query = question
    search_vec = gist_embedding.encode(search_query, convert_to_tensor=True)
    svec = np.array(search_vec).reshape(1, -1)
    svec = np.ascontiguousarray(svec, dtype='float32')
    return svec

# Function to perform RetrievalQA using Hugging Face model
def retrieval_qa(prompt, documents, model, tokenizer, temperature=0.7, max_length=1000):
    input_text = f"{prompt}\n\nDocuments:\n" + "\n".join(documents)
    response = generate_text(input_text, model, tokenizer, max_length=max_length)
    truncated_docs = [doc[:6000] for doc in documents]  # Adjust the length as needed
    return response

# Streamlit UI
st.markdown("<h1 style='color: white; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;'>AeroInsights</h1>", unsafe_allow_html=True)

# User input for base URL
base_url = st.text_input("Enter the base URL:")

# User input for number of pages to scrape
num_pages_to_scrape = st.number_input("Enter the number of pages to scrape:", min_value=1, value=1, step=1)

# Button to trigger scraping
with st.form(key='scrape_form'):
    submit_button = st.form_submit_button("Scrape Reviews")

    if submit_button:
        if not base_url:
            st.warning("Please enter a base URL.")
        else:
            with st.spinner("Scraping in progress..."):
                scrape(base_url, num_pages_to_scrape)
                st.success("Scraping successfully completed.")

# Load text and perform text splitting
reviews_text = load_text()
docs = split_text(reviews_text)

# Initialize Sentence Transformers embeddings
gist_embedding, _list = initialize_sentence_transformers_embeddings(docs)

# Display input fields for airport name and question
st.markdown("<h1 style='color: white; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;'>Airport Analysis</h1>", unsafe_allow_html=True)
with st.form("analyze_form"):
    question = st.text_input("Ask a Question:")
    submit_button = st.form_submit_button("Analyze Airport Reviews")
    st.subheader("Analysis")
    if submit_button and question:
        # Convert user question to embeddings
        svec = convert_question_to_embeddings(question, gist_embedding)

        # Perform similarity search using Faiss
        dim = _list.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(_list)
        distance, I = index.search(svec, k=25)

        # Get similar documents
        lst = [docs[i] for i in I[0]]
        result = retrieval_qa(question, lst, model, tokenizer)
        result_html = f"""<div style="background-color: #ffffff; color: #333333; padding: 10px; border-radius: 5px;">{result}</div>"""
        st.session_state['analysis_result'] = result
    elif submit_button:
        st.warning("Please enter both the airport name and the question.")

if 'analysis_result' in st.session_state:
    result_html = f"""<div style="background-color: #AEC6CF; color: #333333; padding: 10px; border-radius: 5px;">{st.session_state['analysis_result']}</div>"""
    st.markdown(result_html, unsafe_allow_html=True)

# Read the contents of the reviews.txt file with explicit encoding
with open('reviews.txt', 'r', encoding='utf-8') as file:
    reviews_text = file.read()

# Use regular expression to find "Review <number>: Recommended<status>"
matches = re.findall(r'Review (\d+):[\s\S]*?Recommended(.*?)\n', reviews_text)

# Extract review number and 'no' or 'yes' from the found matches
recommendations = [(num, status.strip().lower()) for num, status in matches]

# Separate 'no' and 'yes' recommendations into different lists
no_recommendations = [num for num, status in recommendations if status == 'no']
yes_recommendations = [num for num, status in recommendations if status == 'yes']

def data():
    total_reviews = len(no_recommendations) + len(yes_recommendations)
    positive_reviews = len(yes_recommendations)
    negative_reviews = total_reviews - positive_reviews
    return total_reviews, positive_reviews, negative_reviews

sentiment_pipeline = pipeline("sentiment-analysis")

# Function to read reviews from file
def read_reviews(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        reviews = re.split(r'Review \d+:', content)
        reviews = [review.strip() for review in reviews if review.strip()]
    return reviews

# Classify sentiments of reviews and separate text by sentiment
def classify_and_separate_reviews(reviews):
    positive_reviews = []
    negative_reviews = []
    results = sentiment_pipeline(reviews)
    for review, result in zip(reviews, results):
        label = result['label'].upper()
        if label == 'POSITIVE':
            positive_reviews.append(review)
        elif label == 'NEGATIVE':
            negative_reviews.append(review)
    return positive_reviews, negative_reviews

# Generate and display a word cloud
def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)

# Create and display a bar graph for review sentiment counts
def display_sentiment_counts(positive_reviews, negative_reviews):
    sentiments = ['Positive', 'Negative']
    counts = [len(positive_reviews), len]

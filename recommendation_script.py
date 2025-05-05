# -- coding: utf-8 --
# Medicine Recommendation System (with user input, voice input, and chatbot interface)

# Step 1: Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import speech_recognition as sr  # For voice input
import gradio as gr  # For chatbot interface

# Step 2: Dataset path
dataset = r"finalmedicine_dataset.csv"

# Step 3: Load and preprocess the dataset
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path, on_bad_lines="skip")
        data.columns = data.columns.str.lower()

        required_columns = [
            "name", "category", "dosage form", "strength",
            "manufacturer", "indication", "classification"
        ]
        if not all(col in data.columns for col in required_columns):
            print("Available columns in the dataset:", data.columns)
            raise ValueError(f"The dataset must contain: {required_columns}")

        data.fillna("Unknown", inplace=True)
        return data
    except pd.errors.ParserError as e:
        print("ParserError:", e)
        raise

# Step 4: Check dataset file
if not os.path.exists(dataset):
    raise FileNotFoundError(f"The file '{dataset}' does not exist.")

data = load_dataset(dataset)

# Combine relevant fields into one string
data["text"] = (
    "Name: " + data["name"] + "\n"
    + "Category: " + data["category"] + "\n"
    + "Dosage Form: " + data["dosage form"] + "\n"
    + "Strength: " + data["strength"] + "\n"
    + "Manufacturer: " + data["manufacturer"] + "\n"
    + "Indication: " + data["indication"] + "\n"
    + "Classification: " + data["classification"]
)

# Step 5: Generate embeddings (batched with attention masking)
def generate_embeddings(data, column, tokenizer, model, batch_size=16):
    model.eval()
    embeddings = []

    dataloader = DataLoader(data[column].tolist(), batch_size=batch_size)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Embeddings"):
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
            outputs = model(**inputs)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked_output = outputs.last_hidden_state * attention_mask
            sentence_embeddings = masked_output.sum(dim=1) / attention_mask.sum(dim=1)
            embeddings.extend(sentence_embeddings.cpu().numpy())
    
    return np.array(embeddings)

# Load DistilBERT once
embedding_model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
model = AutoModel.from_pretrained(embedding_model_name)

embeddings = generate_embeddings(data, "text", tokenizer, model)

# Step 6: Recommend medicines
def recommend_medicines(user_query, data, embeddings, tokenizer, model, top_n=5):
    inputs = tokenizer(user_query, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        masked_output = outputs.last_hidden_state * attention_mask
        user_embedding = masked_output.sum(dim=1) / attention_mask.sum(dim=1)

    similarities = cosine_similarity(user_embedding.cpu().numpy(), embeddings)
    top_indices = similarities[0].argsort()[-top_n:][::-1]

    return data.iloc[top_indices]

# Step 7: Side Effects Predictor (Simulation)
def predict_side_effects(medicine_name):
    # Simple simulated side effect data
    side_effects = {
        "Aspirin": ["Nausea", "Vomiting", "Rash"],
        "Ibuprofen": ["Dizziness", "Headache", "Stomach pain"],
        "Paracetamol": ["Liver damage (overuse)", "Nausea"],
        "Amoxicillin": ["Diarrhea", "Rash", "Allergic reaction"],
        # Add more medicines and their side effects here
    }

    return side_effects.get(medicine_name, ["No known side effects"])

# Step 8: Voice Input Functionality
def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service")
            return ""

# Step 9: Main function (for chatbot interface)
def chatbot_interface(query):
    if not query:
        return "Please enter a valid condition."
    recs = recommend_medicines(query, data, embeddings, tokenizer, model, top_n=5)
    output = ""
    for _, row in recs.iterrows():
        output += f"**{row['name']}** (Category: {row['category']})\n"
        se = predict_side_effects(row['name'])
        output += f"Side Effects: {', '.join(se)}\n\n"
    return output

# Step 10: Gradio Interface (Chatbot + Voice Input)
def run_gradio_interface():
    interface = gr.Interface(
        fn=chatbot_interface,
        inputs=gr.Textbox(
            lines=1,
            placeholder="Enter disease/condition, e.g., fever...",
            label="Your Query"
        ),
        outputs=gr.Textbox(
            lines=10,
            label="Recommendations & Side Effects"
        ),
        title="🩺 Medicine Recommendation System",
        description="Type in a disease or condition to get top medicine recommendations along with common side effects."
    )
    interface.launch()

# Run the script
if __name__ == "__main__":
    print("Starting the Medicine Recommendation System...\n")
    
    # Start the Gradio chatbot interface
    run_gradio_interface()

    # To enable voice input, you can uncomment the following line to test voice recognition:
    # query = voice_input()  # This will allow you to input via voice
    # print(f"Voice Input: {query}")

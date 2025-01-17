# -- coding: utf-8 --
# Medicine Recommendation System (with user input)

# Step 1: Install necessary libraries (if running in a new environment)
# !pip install datasets pandas numpy scikit-learn transformers torch

# Step 2: Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Step 3: Define the dataset pathz
dataset = r"script\finalmedicine_dataset.csv"

# Step 4: Load and preprocess the dataset
def load_dataset(file_path):
    try:
        # Load the dataset, skipping problematic lines
        data = pd.read_csv(file_path, on_bad_lines="skip")

        # Standardize column names
        data.columns = data.columns.str.lower()

        # Required columns
        required_columns = [
            "name",
            "category",
            "dosage form",
            "strength",
            "manufacturer",
            "indication",
            "classification",
        ]
        if not all(col in data.columns for col in required_columns):
            print("Available columns in the dataset:", data.columns)
            raise ValueError(
                f"The dataset must contain the following columns: {required_columns}"
            )

        # Fill missing values
        data.fillna("Unknown", inplace=True)
        return data
    except pd.errors.ParserError as e:
        print("ParserError encountered:", e)
        raise

# Check if the file exists
if not os.path.exists(dataset):
    raise FileNotFoundError(f"The file '{dataset}' does not exist.")

data = load_dataset(dataset)

# Combine relevant columns into a single text prompt
data["text"] = (
    "Name: " + data["name"] + "\n"
    + "Category: " + data["category"] + "\n"
    + "Dosage Form: " + data["dosage form"] + "\n"
    + "Strength: " + data["strength"] + "\n"
    + "Manufacturer: " + data["manufacturer"] + "\n"
    + "Indication: " + data["indication"] + "\n"
    + "Classification: " + data["classification"]
)

# Step 5: Generate embeddings for recommendations
def generate_embeddings(data, column, model_name="distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embeddings = []
    for text in data[column]:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        # Take mean of the last hidden states
        sentence_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(sentence_embedding[0])
    return np.array(embeddings)

embeddings = generate_embeddings(data, "text")

# Step 6: Define recommendation function
def recommend_medicines(user_query, data, embeddings, top_n=5):
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Generate user query embedding
    inputs = tokenizer(user_query, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    user_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    # Compute cosine similarity
    similarities = cosine_similarity(user_embedding, embeddings)
    top_indices = similarities[0].argsort()[-top_n:][::-1]

    # Return top recommendations
    return data.iloc[top_indices]

# Step 7: Run the script with user input
def main():
    user_query = input("Enter the disease or condition to get medicine recommendations: ")
    print(f"Generating recommendations for query: {user_query}")

    recommendations = recommend_medicines(user_query, data, embeddings, top_n=5)

    print("\nTop Recommendations:")
    print(recommendations[["name", "category", "indication"]])

# Step 8: (Optional) Fine-tune GPT-2 for enhanced functionality
def fine_tune_gpt2(data):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    hf_dataset = Dataset.from_pandas(data[["text"]])

    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()  # Copy input_ids to labels
        return tokenized

    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=500,
        evaluation_strategy="no",
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained("medicine_recommendation_model")
    tokenizer.save_pretrained("medicine_recommendation_model")

    print("Model fine-tuning complete. Model saved to 'medicine_recommendation_model'.")

if __name__ == "__main__":
    main()
    # Uncomment to run fine-tuning
    # fine_tune_gpt2(data)

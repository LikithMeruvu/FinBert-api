from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import uvicorn

app = FastAPI()

# Load the tokenizer used during fine-tuning
tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')

# Load the fine-tuned model
model_path = "likith123/SSAF-FinBert"
model = AutoModelForSequenceClassification.from_pretrained(model_path)

class Item(BaseModel):
    text: str
    title: str

# Define a function for sentiment prediction
async def predict_sentiment(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted probabilities for each class
    predicted_probs = torch.softmax(outputs.logits, dim=1).squeeze().tolist()

    return predicted_probs

# Cache the loaded model
loaded_model = None

def get_model():
    global loaded_model
    if loaded_model is None:
        loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return loaded_model

@app.get("/")
def index():
    return {"Sentiment Analysis"}

# Sentiment analysis endpoint
@app.post("/sentiment-analysis/")
async def analyze_sentiment(item: Item):
    input_text = item.title + " " + item.text
    model = get_model()
    predicted_probs = await predict_sentiment(input_text)
    # Map predicted labels to human-readable sentiment
    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    predicted_sentiment = sentiment_mapping[predicted_probs.index(max(predicted_probs))]
    # Create response body with keys for each sentiment class
    response_body = {
        "predicted_sentiment": predicted_sentiment,
        "predicted_probabilities": {
            "Negative": predicted_probs[0],
            "Neutral": predicted_probs[1],
            "Positive": predicted_probs[2]
        }
    }
    return response_body

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

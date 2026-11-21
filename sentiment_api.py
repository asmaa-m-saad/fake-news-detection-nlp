import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import os


# 1. Load Fake News Model
FAKE_MODEL_PATH = os.getenv("FAKE_MODEL_PATH", "saved_fakenews_model")

fake_tokenizer = None
fake_model = None
try:
    fake_tokenizer = BertTokenizer.from_pretrained(FAKE_MODEL_PATH)
    fake_model = BertForSequenceClassification.from_pretrained(FAKE_MODEL_PATH)
    fake_model.eval()
except Exception as e:
    print(
        f"\nWARNING: could not load the fake news model from '{FAKE_MODEL_PATH}'.\n"
        "Run the BERT training cells in the notebook first (they save the model to 'saved_fakenews_model'),\n"
        "or set the FAKE_MODEL_PATH environment variable. The app will still start, but fake news\n"
        "prediction will be unavailable (sentiment analysis will still work).\n"
        f"Details: {e}\n"
    )

def predict_fake_news(text: str):
    if fake_model is None:
        return "Unavailable (fake news model not found, see README)"
    inputs = fake_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = fake_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Real" if prediction == 1 else "Fake"


# 2. Load Sentiment Model
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
sent_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
sent_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
sent_model.eval()

labels = ["Negative", "Neutral", "Positive"]

def predict_sentiment(text: str):
    inputs = sent_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512  # Force proper truncation
    )
    # Drop token_type_ids if they exist
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        outputs = sent_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    sentiment = labels[pred]  # 0 = Negative, 1 = Neutral, 2 = Positive
    return sentiment, round(confidence * 100, 2)



# 3. FastAPI App
app = FastAPI()
# index.html sits in the same folder as this file
templates = Jinja2Templates(directory=".")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, text: str = Form(...)):
    fake_result = predict_fake_news(text)
    sentiment_result, confidence = predict_sentiment(text)
    result = {
        "fake_news": fake_result,
        "sentiment": sentiment_result,
        "confidence": confidence
    }
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "text": text})


# 4. Debug Routes 
@app.get("/__plain_test__")
async def plain_test():
    return HTMLResponse("<h1>FastAPI is running ✅</h1>")

@app.get("/__debug_info__")
async def debug_info():
    routes = [{"path": r.path, "methods": list(r.methods)} for r in app.routes]
    return {
        "cwd": os.getcwd(),
        "files_in_cwd": sorted(os.listdir(".")),
        "templates_exists": os.path.isdir("."),
        "templates_files": sorted(os.listdir(".")) if os.path.isdir(".") else [],
        "registered_routes": routes,
    }

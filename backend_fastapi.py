from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model = tf.keras.models.load_model("Cloud_AIML_IT_service_ticket_model.keras")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# FastAPI instance
app = FastAPI()

# Define request model
class TicketRequest(BaseModel):
    description: str
    severity: int
    priority: int

# Prediction function
def predict_ticket(ticket: TicketRequest):
    # Preprocess text
    sequence = tokenizer.texts_to_sequences([ticket.description])
    padded_sequence = pad_sequences(sequence, maxlen=50, padding="post")

    # Prepare input
    model_input = [np.array(padded_sequence), np.array([[ticket.severity, ticket.priority]])]

    # Get predictions
    category_pred, resolution_pred = model.predict(model_input)
    predicted_category = np.argmax(category_pred)
    predicted_days = int(round(resolution_pred[0][0]))

    return {"predicted_category": int(predicted_category), "predicted_resolution_days": predicted_days}

# API endpoint
@app.post("/predict")
def get_prediction(ticket: TicketRequest):
    return predict_ticket(ticket)

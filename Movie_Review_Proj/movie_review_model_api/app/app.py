
import json
import numpy as np
import pandas as pd

from typing import Any
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI
from fastapi.params import Body
from movie_review_model import __version__ as model_version
from movie_review_model.predict import predict


from app import __version__, schemas
from app.config import settings

app = FastAPI()
api_router = APIRouter()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict_sentiment(review_text: str = Body(...)):
    sentiment = predict(review_text)
    return sentiment

@app.get("/predict/{review_text}")
async def predict(review_text: str):
    sentiment = predict(review_text)
    return sentiment
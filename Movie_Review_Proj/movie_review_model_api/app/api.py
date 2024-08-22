
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


from app import __version__
# from app.config import settings


api_router = APIRouter()

@api_router.get("/")
def read_root():
    return {"Hello": "World"}


@api_router.post("/predict")
def predict_sentiment(review_text: str = Body(...)):
    sentiment = predict(review_text)
    return sentiment

# @api_router.get("/predict/{review_text}")
# async def predict(review_text: str):
#     sentiment = predict(review_text)
#     return sentiment
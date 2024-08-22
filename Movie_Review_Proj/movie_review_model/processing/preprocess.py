"""Pre-processing"""
from pathlib import Path
import sys

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from movie_review_model.config.core import config

from sklearn.base import BaseEstimator, TransformerMixin
from transformers import BertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import  Dataset, DataLoader

###  Pre-Pipeline Preparation ###

class TokenizerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,  max_length = config.model.max_length):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=False)
        self.max_length = max_length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure that input X is a list of strings
        if not isinstance(X, list) or not all(isinstance(text, str) for text in X):
            raise ValueError("Input to tokenizer must be a list of strings.")
        # Tokenize the input text and return input_ids
        tokenized = self.tokenizer(
            X,
            return_tensors="np", # Use "np" to return NumPy arrays
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        # Only return the input_ids
        return tokenized['input_ids']

    def tokenize_text(self, texts, max_length=256):
        encoding = self.tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return encoding

class MovieReviewDataset(Dataset):
    """
    Moview review dataset class.
    """

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {'text': self.texts[idx]}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class SentimentClassifier:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def fit(self, X, y):
        dataset = MovieReviewDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        self.model.to(self.device)

        for epoch in range(config.model.epochs):
            self.model.train()
            for batch in dataloader:
                # batch = {k: v.to(self.device) for k, v in batch.items()}
                # outputs = self.model(**batch)
                input_ids = batch['text'].to(self.device)
                attention_mask = (input_ids != 0).to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs.logits, batch['labels'].to(self.device))
                if loss is not None:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    print("Loss calculation failed. Skipping backward pass.")

                print(f"processing batches for epoch:{epoch + 1}  ...")
            print(f"\nEpoch {epoch + 1} complete. Loss: {loss.item()}")

        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**X)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
        return predictions

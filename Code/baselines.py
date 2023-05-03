import numpy as np
from tqdm import tqdm
import os
import requests
from multiprocessing.pool import ThreadPool
from math import ceil

class PXDetector:
    def __init__(self):
        pass
    
    def score(self, data, model, batch_size=8, **kwargs):
        print("Scoring")
        n_samples = len(data)
        scores = []
        for batch in tqdm(range(ceil(n_samples / batch_size)), desc=f"Computing PX"):
            original_text = data[batch * batch_size:(batch + 1) * batch_size]
            loglikelihood = model.get_batch_loglikelihood(original_text)
            scores.append(loglikelihood)
        scores = np.concatenate(scores)
        return scores

class GPT0:
    def __init__(self, api_key=""):
        self.api_key = api_key
        self.base_url = 'https://api.gptzero.me/v2/predict'
        pass
    
    def score(self, data, model, batch_size=8, **kwargs):
        print("Scoring")
        n_samples = len(data)
        scores = []
        for batch in tqdm(range(n_samples // batch_size), desc=f"Computing PX"):
            original_text = data[batch * batch_size:(batch + 1) * batch_size]
            loglikelihood = model.get_batch_loglikelihood(original_text)
            scores.append(loglikelihood)
        scores = np.concatenate(scores)
        return scores


    def score(self, documents, n_p=16):
        pool = ThreadPool(n_p)
        out_scores = pool.map(self.text_predict, documents)
        
        
        
    def text_predict(self, document):
        url = f'{self.base_url}/text'
        headers = {
            'accept': 'application/json',
            'X-Api-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        data = {
            'document': document
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()

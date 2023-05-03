from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import numpy as np
import torch
from scipy.special import softmax



class HFDetector:
    def __init__(self, model_name="Hello-SimpleAI/chatgpt-detector-roberta" , cache_dir="/dfs/scratch1/merty/.cache", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, output_hidden_states=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir, output_hidden_states=True)
        self.model = self.model.to(device)
        self.model = self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def score(self, texts, model, batch_size=8):
        all_scores = []
        for batch in tqdm(range(len(texts) // batch_size), desc=f"Running ChatGPT Detector"):
            original_text = texts[batch * batch_size:(batch + 1) * batch_size]
            inputs = self.tokenizer(original_text, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            logits = outputs["logits"].detach().cpu().numpy()
            probs = softmax(logits, axis=1)
            all_scores.append(probs[:, 1:])
        all_scores = np.concatenate(all_scores, axis=0)
        return all_scores




    
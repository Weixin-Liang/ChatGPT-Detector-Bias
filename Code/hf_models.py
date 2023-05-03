import torch
from multiprocessing.pool import ThreadPool
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F


class HFModel:
    def __init__(self, model_name="gpt2" , cache_dir="/dfs/scratch1/merty/.cache", device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = self.model.to(device)
        self.model = self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def sample(self, texts, prompt_tokens=30, min_words=55, do_top_p=True, min_length=150):
        all_encoded = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}
        
        decoded = ['' for _ in range(len(texts))]

        # sample from the model until we get a sample with at least min_words words for each example
        # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
        tries = 0
        while (m := min(len(x.split()) for x in decoded)) < min_words:
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

            sampling_kwargs = {}
            if do_top_p:
                sampling_kwargs['top_p'] = 0.9
            
            outputs = self.model.generate(**all_encoded, min_length=min_length, max_length=200, do_sample=True, **sampling_kwargs, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            tries += 1
        return decoded
    
    @torch.no_grad()
    def get_batch_loglikelihood(self, texts):
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        labels = tokenized.input_ids
        outputs = self.model(**tokenized, labels=labels)
        logits = outputs.logits.cpu()
        labels = labels.cpu()
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")
        ll_per_sample = -loss.view(shift_logits.shape[0], shift_logits.shape[1])
        nonpad_per_row = (shift_labels != -100).sum(dim=1)
        ll_per_sample = ll_per_sample.sum(dim=1)/nonpad_per_row
        return ll_per_sample.cpu().numpy()
    


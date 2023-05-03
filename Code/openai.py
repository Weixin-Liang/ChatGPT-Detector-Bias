import openai
import numpy as np
from transformers import AutoTokenizer
from multiprocessing.pool import ThreadPool


class OpenAIModel:
    def __init__(self, model_name="text-davinci-003", cache_dir="/dfs/scratch1/merty/.cache", device="cuda",
                top_p=0.96, do_top_p=True):
        self.openai_model = model_name
        openai.api_key =  '<Your API Key>' 
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.top_p = top_p
        self.do_top_p = do_top_p
    
    def _sample_api(self, text):
        # sample from the openai model
        kwargs = {"engine": self.openai_model, "max_tokens": 200 }
        if self.do_top_p:
            kwargs['top_p'] = self.top_p
        
        r = openai.Completion.create(prompt=f"{text}", **kwargs)
        return text + r['choices'][0].text

    def sample(self, texts, prompt_tokens=30, min_words=55, do_top_p=True, min_length=150):
        all_encoded = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}
        
        prefixes = self.tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
        pool = ThreadPool(8)
        decoded = pool.map(self._sample_api, prefixes)
        return decoded
    
    def get_ll(self, text):
        kwargs = { "engine": self.openai_model, "temperature": 0, "max_tokens": 0, "echo": True, "logprobs": 0}
        r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
        result = r['choices'][0]
        tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]
        return np.mean(logprobs)
    
    def get_batch_loglikelihood(self, texts):
        # get the loglikelihood of each text in the batch
        ll_per_sample = np.array(ThreadPool(8).map(self.get_ll, texts))
        return ll_per_sample
    
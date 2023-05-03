import os
import torch
import json
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import get_model
from detectors import get_detector

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2-xl")
    parser.add_argument("--dataset_name", type=str, default="toefl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--detector_name", type=str, default="detectgpt")
    parser.add_argument("--cache_dir", type=str, default="/dfs/scratch1/merty/.cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root-folder", type=str, default="./project_data/")
    return parser.parse_args()


args = config()    
model = get_model(args.model_name)
detector = get_detector(args.detector_name)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

import gdown
import os
import glob
import pandas as pd

def pubs_dataset(cache_dir="/dfs/scratch2/merty/", first_k=500):
    if not os.path.exists(os.path.join(cache_dir, "ScientificDiscovery/")):
        raise ValueError()
    
    all_csvs = glob.glob(os.path.join(cache_dir, "ScientificDiscovery/", "*.csv"))
    
    dfs = [pd.read_csv(csv)[['abstract', 'venue']] for csv in all_csvs[:1]]
    concatted = pd.concat(dfs).iloc[:first_k]
    abstracts = concatted['abstract'].tolist()
    venues = concatted['venue'].tolist()
    return concatted, abstracts, venues


class DetectionDataset:
    def __init__(self, name, root="/afs/cs.stanford.edu/u/merty/projects/chatgpt-detector-eval/project_data/"):
        self.data_path = os.path.join(root, name)
        self.name = json.load(open(os.path.join(self.data_path, "name.json")))
        self.data = json.load(open(os.path.join(self.data_path, "data.json")))
        self.results = None
        self.collect_results()
    def __len__(self):
        return len(self.data)
    def describe(self):
        return f"{self.name['name']} Mean Length: {np.mean([len(t['document'].split(' ')) for t in self.data])}"
    def __getitem__(self, idx):
        return self.data[idx]["document"]
    def collect_results(self):
        results_files = [f for f in os.listdir(self.data_path) if f not in ["name.json", "data.json"]]
        print(results_files)
        results_records = [json.load(open(os.path.join(self.data_path, f))) for f in results_files]
        detector_names = [f.split(".")[0] for f in results_files]
        results_records = {n: {"Detector": n, "Scores": r} for n,r in zip(detector_names, results_records)}
        self.results = results_records
        return results_records
    def get_results(self, detector_name):
        if detector_name not in self.results:
            return None
        else:
            return self.results[detector_name]
    def save_result(self, detector_name, scores):
        with open(os.path.join(self.data_path, f"{detector_name}.json"), "w") as f:
            json.dump(scores, f)


root_folder = args.root_folder
for folder in os.listdir(root_folder):
    ds = DetectionDataset(foler)
    print(ds[0])
    print(ds.describe())
    results = ds.collect_results()
    print(results.keys())
    print()

    detector_results = ds.get_results(args.detector_name)
    if detector_results is None:

        texts = [t["document"] for t in ds.data]
        print(ds.describe())
        dataset_scores = detector.score(texts, model, batch_size=args.batch_size)
        all_results = []
        for d, s in zip(texts, dataset_scores):
            all_results.append({"document": d, "score": float(s)})

        ds.save_result(f"{args.detector_name}-{args.model_name}", all_results)
        print("Saved")
    
    else:
        print("Already exists: {}".format(detector_results))
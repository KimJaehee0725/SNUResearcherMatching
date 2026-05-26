import os
import torch
import numpy as np

from sentence_transformers import SentenceTransformer, util
from data.data_controller import load_data_samples

def _compute_mrr_and_recall(similarity_scores, top_k):
    reciprocal_ranks = []
    recall_at_k = []
    
    for idx, scores in enumerate(similarity_scores):
        sorted_indices = scores.argsort(descending=True)
        
        rank = (sorted_indices == idx).nonzero(as_tuple=True)[0]
        if len(rank) > 0:
            reciprocal_ranks.append(1 / (rank[0].item() + 1))
            recall_at_k.append(1 if rank[0].item() < top_k else 0)
    
    return np.mean(reciprocal_ranks), np.mean(recall_at_k)

def _evaluate_model(config, model, queries_flat, pos_docs_flat, output_path, mode=None):
    # TODO: Implement the single evaluation run w/ faiss
    # Temporary code without faiss shows below
    
    with torch.no_grad():
        query_embeddings = model.encode(queries_flat, convert_to_tensor=True)
        pos_embeddings = model.encode(pos_docs_flat, convert_to_tensor=True)
        
        similarity_scores = util.pytorch_cos_sim(query_embeddings, pos_embeddings)
    
    mrr, recall_k = _compute_mrr_and_recall(similarity_scores, top_k=config["top_k"])
    
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Recall@{config['top_k']}: {recall_k:.4f}")
    if os.environ.get("LOCAL_RANK", "0") == "0":
        with open(f'{config["output_path"]}/scores.txt', "w") as f:
            f.write(f"Mean Reciprocal Rank (MRR): {mrr:.4f}\n")
            f.write(f"Recall@{config['top_k']}: {recall_k:.4f}\n")
    
def evaluate_model(config, mode=None):
    # TODO: Implement the whole evaluation logic w/ faiss
    # Temporary code without faiss shows below
    
    samples = load_data_samples(config["test_file"], train=False)
    
    queries_flat = []
    pos_docs_flat = []
    for query_list, pos_doc in samples:
        for query in query_list:
            queries_flat.append(query)
            pos_docs_flat.append(pos_doc)
            
    if mode=='scratch':
        # Evaluate only the scratch once
        model = SentenceTransformer(config["model_name"], device=config["device"], cache_folder='./models')
        _evaluate_model(config, model, queries_flat, pos_docs_flat, config["output_path"], mode)
    elif mode in ['evaluate', 'both']:
        # Evaluate every checkpoint in the output path
        # Checkpoints are in the format of 'checkpoint-<step>' inside the config["output_path"]
        
        # Get all the checkpoint paths
        checkpoint_paths = []
        for file in os.listdir(config["output_path"]):
            if file.startswith("checkpoint-"):
                checkpoint_paths.append(f"{config['output_path']}/{file}")
                
        # Run evaluation for each checkpoint
        for checkpoint_path in checkpoint_paths:
            model = SentenceTransformer(checkpoint_path, device=config["device"])
            _evaluate_model(config, model, queries_flat, pos_docs_flat, checkpoint_path, mode)    

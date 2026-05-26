import os
import faiss
import time
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Set, Optional

PROJECT_TEMPLATE = """
Title: {title}
Major: {major}
"""

RESEARCH_TEMPLATE = """
Title: {title}
Major: {major}
"""

RESEARCH_TEMPLATE_W_ABSTRACT = """
Title: {title}
Major: {major}
Abstract: {abstract}
"""

def load_data(
        corpus_path: str, 
        type: str = 'research'
        ) -> List[Dict]:
    """
    Loads a corpus from a CSV file.

    Args:
        corpus_path (str): The path to the corpus CSV file.
        type (str, optional): The type of corpus, either 'research' or 'project'. Defaults to 'research'.

    Returns:
        List[Dict]: A list of dictionaries representing the corpus.
    """
    # load original corpus w/ metadata
    if type == 'project':
        corpus = pd.read_csv(corpus_path)
    elif type == 'research':
        corpus = pd.read_csv(corpus_path)
    return corpus


def append_template(
        corpus: pd.DataFrame,
        format = 'research',
        start_idx = 0
        ) -> Dict[int, str]:
    """
    Applies a template to each document in the corpus and assigns a unique ID.

    Args:
        corpus (pd.DataFrame): The input corpus as a pandas DataFrame.
        format (str, optional): The format of the corpus, either 'research' or 'project'. Defaults to 'research'.
        start_idx (int, optional): The starting index for the document IDs. Defaults to 0.

    Returns:
        Dict[int, str]: A dictionary mapping a unique ID to the templated document and its metadata.
    """
        
    def _preprocess_research(research):
        if research['abstract'] == 'nan':
            templated =  RESEARCH_TEMPLATE.format(
                title=research['title'],
                major=research['department']
            )
            return (research['doc_id'], templated)
        else: 
            templated = RESEARCH_TEMPLATE_W_ABSTRACT.format(
                title=research['title'],
                major=research['department'],
                abstract=research['abstract']
            )
            return (research['doc_id'], templated)
    
    def _preprocess_project(project):
        templated =  PROJECT_TEMPLATE.format(
            title=project['과제명'],
            major=project['소속부서']
        )

        return (project['doc_id'], templated)
    
    if format == 'research':
        preprocess_fn = _preprocess_research
    elif format == 'project':
        preprocess_fn = _preprocess_project

    metadata = []
    for idx, (_, row) in enumerate(corpus.iterrows()):
        if format == 'research':
            metadata.append({
                'prof': row['korean_name'],
                'unit': row['department'],
                'title': row['title'],
                'id': idx + start_idx,
                'type': 'research'
            })
        elif format == 'project':
            metadata.append({
                'prof': row['연구책임자'],
                'unit': row['소속부서'],
                'title': row['과제명'],
                'id': idx + start_idx,
                'type': 'project'
            })
    
    corpus = corpus.apply(lambda x: preprocess_fn(x), axis=1)
    corpus = {int(id) + start_idx: {'document': doc, 'metadata': meta_row} for (id, doc), meta_row in zip(corpus, metadata)}
    return corpus


def set_faiss(corpus_embeddings, corpus_ids):
    """
    Creates a FAISS index from corpus embeddings and IDs.

    Args:
        corpus_embeddings: The embeddings of the corpus documents.
        corpus_ids: A list of unique IDs for each document in the corpus.

    Returns:
        faiss.Index: A FAISS index populated with the documents.
    """
    corpus_embeddings = corpus_embeddings.cpu()
    index_flat = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index = faiss.IndexIDMap(index_flat)
    index.add_with_ids(corpus_embeddings.numpy(), corpus_ids)
    return index


def get_model_and_corpus_with_index(
        device = 'cuda:0',
        model_path = '/workspace/data/SNU_Eng_Retrieval/results/BAAI__bge-m3/v2.13_bge-m3/checkpoint-2060',
        research_corpus_path = '/workspace/data/SNU_Eng_Retrieval/translation_data/research_translation_final.csv',
        project_corpus_path = '/workspace/data/SNU_Eng_Retrieval/translation_data/project_translation_final.csv',
        research_new_prof_path = './new_researchers/research_new_prof_10.txt',
        project_new_prof_path = './new_researchers/project_new_prof_10.txt',
        index_path = '/workspace/data/SNU_Eng_Retrieval/demo/ver1'
        ):
    """
    Initializes and returns the model, corpora, and FAISS indices.

    If a FAISS index does not exist at the specified path, it will be created
    by encoding the corpus documents and saved for future use.

    Args:
        device (str, optional): The device to run the model on. Defaults to 'cuda:0'.
        model_path (str, optional): Path to the SentenceTransformer model.
        research_corpus_path (str, optional): Path to the research corpus CSV.
        project_corpus_path (str, optional): Path to the project corpus CSV.
        research_new_prof_path (str, optional): Path to the list of new researchers (research).
        project_new_prof_path (str, optional): Path to the list of new researchers (project).
        index_path (str, optional): Directory to save/load FAISS indices.

    Returns:
        tuple: A tuple containing:
            - model: The loaded SentenceTransformer model.
            - full_corpus: The combined research and project corpus.
            - full_index: The FAISS index for the full corpus.
            - subset_corpus: The corpus of new researchers.
            - subset_index: The FAISS index for the subset corpus.
    """
    # load model
    model = SentenceTransformer(model_path, device)
    model.eval()
    
    # load data
    research_data = load_data(research_corpus_path, type='research')
    research_corpus = append_template(research_data, format='research')
    
    project_data = load_data(project_corpus_path, type='project')
    project_corpus = append_template(project_data, format='project', start_idx= len(research_corpus))
    
    # aggregate data
    full_corpus = {**research_corpus, **project_corpus} # {faiss index id: {'document': ... , 'metadata': {'prof': ... , 'unit': 소속학과 , 'title': ... , 'id': unique id , 'type': ... }}}


    # sample 1000 documents for debugging
    # corpus = {k: v for k, v in corpus.items() if k < 1000}
    # load or set up faiss index
    full_db_index_path = os.path.join(index_path, 'full.faiss')
    if os.path.exists(full_db_index_path):
        full_index = faiss.read_index(full_db_index_path)
    else:        
        # encode corpus
        documents = [doc['document'] for doc in full_corpus.values()]
        corpus_embeddings = model.encode(documents, convert_to_tensor=True, show_progress_bar=True, batch_size=128)

        # set up faiss
        doc_ids = list(full_corpus.keys())
        full_index = set_faiss(corpus_embeddings, doc_ids)
        faiss.write_index(full_index, full_db_index_path)
    
    # load new professors depend on research or project
    with open(research_new_prof_path, 'r') as f:
        research_new_prof = f.read().splitlines()
    with open(project_new_prof_path, 'r') as f:
        project_new_prof = f.read().splitlines()

    new_prof = set(research_new_prof) | set(project_new_prof)
    subset_corpus = {k: v for k, v in full_corpus.items() if v['metadata']['prof'] in new_prof}

    # set up faiss index for subset corpus
    subset_index_path = os.path.join(index_path, 'subset.faiss')
    if os.path.exists(subset_index_path):
        subset_index = faiss.read_index(subset_index_path)
    else:
        subset_documents = [doc['document'] for doc in subset_corpus.values()]
        subset_corpus_embeddings = model.encode(subset_documents, convert_to_tensor=True, show_progress_bar=True, batch_size=128)
        subset_doc_ids = list(subset_corpus.keys())
        subset_index = set_faiss(subset_corpus_embeddings, subset_doc_ids)
        faiss.write_index(subset_index, subset_index_path)

    return model, full_corpus, full_index, subset_corpus, subset_index


def _find_doc_by_ids(
        retrieved: Dict, 
        docs: List[Dict]
        ):
    """
    Finds and formats document details based on retrieved IDs.

    Args:
        retrieved (Dict): A dictionary containing 'corpus_id' and 'score' lists from FAISS search.
        docs (List[Dict]): The full corpus, a dictionary mapping doc_id to document content and metadata.

    Returns:
        List[Dict]: A list of formatted documents, each including rank, score, and metadata.
    """
    scores = retrieved['score']
    ids = retrieved['corpus_id']

    return [
            {
                'doc_id': int(id),
                'rank' : int(rank + 1),
                'score': float(score),
                'prof': docs[id]['metadata']['prof'],
                'title': docs[id]['metadata']['title'],
                'department': docs[id]['metadata']['unit'],
                'type': docs[id]['metadata']['type']
            } for rank, (score, id) in enumerate(zip(scores, ids))
        ]


def _filtered_by_professor(
        user_q: str,
        index: faiss.IndexFlatL2,
        corpus: dict,
        model: SentenceTransformer,
        large_k: int,
        k: int,
        p: int,
        excluded_profs: Optional[Set[str]] = None
        ) -> dict:
    """
    Searches the index, retrieves documents, and filters them by professor.

    This function first performs a broad search to find `large_k` documents,
    then processes the results to identify the top `k` professors and returns
    up to `p` documents for each of them.

    Args:
        user_q (str): The user's search query.
        index (faiss.IndexFlatL2): The FAISS index to search.
        corpus (dict): The corpus containing document metadata.
        model (SentenceTransformer): The model to encode the query.
        large_k (int): The initial number of documents to retrieve from FAISS.
        k (int): The number of top professors to return.
        p (int): The maximum number of documents to return per professor.
        excluded_profs (Optional[Set[str]], optional): A set of professor names to exclude from the results. Defaults to None.

    Returns:
        dict: A dictionary mapping professor names to a list of their relevant documents.
    """
    
    # Vectorise user_q abd retrieve large_k similar documents
    query_embedding = model.encode([user_q], convert_to_tensor=True)
    top_large_k_sims, top_large_k_ids = index.search(query_embedding.cpu().numpy(), large_k)
    
    top_large_k_retrieved_doc_ids = [{'score': sim, 'corpus_id': id} for rank, (sim, id) in enumerate(zip(top_large_k_sims, top_large_k_ids))]
    top_large_k_retrieved_docs = _find_doc_by_ids(top_large_k_retrieved_doc_ids[0], corpus)
    
    # filtering
    prof_docs_map = {}
    prof_counts = {}
    top_k_professors = []
    prof_set = set()

    # Calculate top_k_professors and make prof_docs_map
    for doc in top_large_k_retrieved_docs:
        prof = doc['prof']
        
        # 제외 목록에 있는 교수는 건너뜁니다.
        if excluded_profs and prof in excluded_profs:
            continue

        # top_k_professors 추출
        if prof not in prof_set:
            if len(top_k_professors) < k:
                top_k_professors.append(prof)
                prof_set.add(prof)

        # Add only up to p documents per professors
        if prof in prof_set:
            if prof not in prof_docs_map:
                prof_docs_map[prof] = []
                prof_counts[prof] = 0
            if prof_counts[prof] < p:
                prof_docs_map[prof].append(doc)
                prof_counts[prof] += 1

    return prof_docs_map


def search_full_db(
        user_q: str,
        index: faiss.IndexFlatL2,
        corpus: dict, 
        model: SentenceTransformer, 
        large_k: int, 
        k_full: int, 
        num_docs_full: int
        ):
    """
    Searches the full database for professors and documents matching the query.

    Args:
        user_q (str): The user's search query.
        index (faiss.IndexFlatL2): The FAISS index for the full corpus.
        corpus (dict): The full corpus of documents.
        model (SentenceTransformer): The sentence embedding model.
        large_k (int): The initial number of documents to retrieve for broad search.
        k_full (int): The target number of top professors to return.
        num_docs_full (int): The maximum number of documents per professor.

    Returns:
        dict: A dictionary mapping professor names to a list of their relevant documents.
    """
    
    # Retrieve top k professors and their documents(up to p) based on user query
    prof_docs_map = _filtered_by_professor(user_q, index, corpus, model, large_k, k_full, num_docs_full)

    # Convert the results into a dictionary with professor names as keys
    result = {}
    for rank_idx in range(k_full):
        name = list(prof_docs_map.keys())[rank_idx]
        contents = prof_docs_map[name]
        
        entry = {}
        # Add department
        if len(contents) > 0:
            entry['department'] = contents[0]['department']
            
        # Add documents and scores
        for j in range(len(contents)): # if prof has less than p documents, return only those
            docs = contents[j]
            entry[f'document{j+1}'] = docs['title']
            entry[f'score{j+1}'] = docs['score']
            entry[f'type{j+1}'] = docs['type']
            
        result[name] = entry
    
    return result


def search_subset_db(
        user_q: str,
        index: faiss.IndexFlatL2,
        corpus: dict, 
        model: SentenceTransformer, 
        large_k: int, 
        k_subset: int, 
        num_docs_subset: int
        ):
    """
    Searches the subset database (new professors) for documents matching the query.

    Args:
        user_q (str): The user's search query.
        index (faiss.IndexFlatL2): The FAISS index for the subset corpus.
        corpus (dict): The subset corpus of documents.
        model (SentenceTransformer): The sentence embedding model.
        large_k (int): The initial number of documents to retrieve for broad search.
        k_subset (int): The target number of top professors to return.
        num_docs_subset (int): The maximum number of documents per professor.

    Returns:
        dict: A dictionary mapping professor names to a list of their relevant documents.
    """
    
    # Retrieve top k professors and their documents(up to b) based on user query
    prof_docs_map = _filtered_by_professor(user_q, index, corpus, model, large_k, k_subset, num_docs_subset)
    
    # Convert the results into a dictionary with professor names as keys
    result = {}
    for prof in range(k_subset):
        name = list(prof_docs_map.keys())[prof]
        contents = prof_docs_map[name]
        
        entry = {}
        # Add department
        if len(contents) > 0:
            entry['department'] = contents[0]['department']
            
        # Add documents and scores
        for j in range(len(contents)): # if prof has less than b documents, return only those
            docs = contents[j]
            entry[f'document{j+1}'] = docs['title']
            entry[f'score{j+1}'] = docs['score']
            entry[f'type{j+1}'] = docs['type']
            
        result[name] = entry    
    
    return result


def find_similar_professors(
        document: str,
        index: faiss.IndexFlatL2,
        corpus: dict, 
        model: SentenceTransformer, 
        large_k: int, 
        k_similar: int, 
        num_docs_similar: int,
        excluded_profs: Optional[Set[str]] = None
        ):
    """
    Finds professors who have conducted similar research to the given document.

    This function searches for the top professors whose documents are most similar
    to a given input document and retrieves a specified number of documents for each
    matched professor.

    Args:
        document (str): A document to compare against the corpus.
        index (faiss.IndexFlatL2): The FAISS index to search.
        corpus (dict): The corpus containing document metadata.
        model (SentenceTransformer): The model to encode the document text.
        large_k (int): The initial number of documents to retrieve.
        k_similar (int): The number of most relevant professors to return.
        num_docs_similar (int): The maximum number of top-matching documents to return per professor.
        excluded_profs (Optional[Set[str]], optional): A set of professor names to exclude from the results.

    Returns:
        dict: A dictionary where each key is a professor's name and each value is
              a nested dictionary containing the department, document titles, scores, and types.
    """
    
    # Retrieve top k professors and their documents(up to b) based on user query
    prof_docs_map = _filtered_by_professor(document, index, corpus, model, large_k, k_similar, num_docs_similar, excluded_profs=excluded_profs)
    
    # Convert the results into a dictionary with professor names as keys
    result = {}
    if not prof_docs_map:
        return result
        
    for prof in range(min(k_similar, len(prof_docs_map))):
        name = list(prof_docs_map.keys())[prof]
        contents = prof_docs_map[name]
        
        entry = {}
        # Add department
        if len(contents) > 0:
            entry['department'] = contents[0]['department']
            
        # Add documents and scores
        for j in range(len(contents)): # if prof has less than b documents, return only those
            docs = contents[j]
            entry[f'document{j+1}'] = docs['title']
            entry[f'score{j+1}'] = docs['score']
            entry[f'type{j+1}'] = docs['type']
            
        result[name] = entry
    
    return result
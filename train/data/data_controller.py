from __future__ import annotations

import re
import random
import jsonlines

from torch.utils.data import Sampler
from datasets import Dataset
from tqdm import tqdm

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

import torch

def _data_sanity_check(sample):
    """
    Checks if a sample dictionary has the required keys for processing.

    This function verifies that a given sample, loaded from a JSONL file,
    contains the necessary nested structure and keys for creating training
    examples. It prints warnings if keys are missing.

    Data should be able to be parsed like follows:
    
    sample_processed = {
            "과제명": sample["과제명"] or sample["논문 제목"],
            "소속학과": sample["소속학과"],
            "간단 키워드": sample["키워드 기반 쿼리"]["매우 간단한 버전"],
            "보통 키워드": sample["키워드 기반 쿼리"]["보통 수준의 구체화"],
            "구체 키워드": sample["키워드 기반 쿼리"]["매우 구체적인 버전"],
            "간단 문장": sample["문장형 쿼리"]["매우 간단한 버전"],
            "보통 문장": sample["문장형 쿼리"]["보통 수준의 구체화"],
            "구체 문장": sample["문장형 쿼리"]["매우 구체적인 버전"],
            "간단 질문": sample["질문형 쿼리"]["매우 간단한 버전"],
            "보통 질문": sample["질문형 쿼리"]["보통 수준의 구체화"],
            "구체 질문": sample["질문형 쿼리"]["매우 구체적인 버전"],
        }
    
    Args:
        sample (dict): Single data point from .jsonl file
        
    Returns:
        bool: True if sample has all required keys, False otherwise.
    """
    
    if not isinstance(sample, dict):
        print("Warning: sample is not a dictionary.")
        return False

    required_keys = [["과제명", "논문 제목"], "소속학과", "키워드 기반 쿼리", "문장형 쿼리", "질문형 쿼리"]
    required_subkeys = ["매우 간단한 버전", "보통 수준의 구체화", "매우 구체적인 버전"]
    
    for key in required_keys:
        if isinstance(key, list):
            if not any(k in sample for k in key):
                print(f"Warning: Missing top-level key: {key}")
                print(f"Keys found: {sample.keys()}")
                return False
        else:
            if key not in sample:
                print(f"Warning: Missing top-level key: '{key}'")
                print(f"Keys found: {sample.keys()}")
                return False

    for category in ["키워드 기반 쿼리", "문장형 쿼리", "질문형 쿼리"]:
        sub_dict = sample.get(category)
        if not isinstance(sub_dict, dict):
            print(f"Warning: '{category}' should be a dictionary.")
            return False
        for subkey in required_subkeys:
            if subkey not in sub_dict:
                print(f"Warning: Missing key '{subkey}' in '{category}'")
                print(f"Keys found in '{category}': {sub_dict.keys()}")
                return False

    return True

def _preprocess_doc(item):
    """
    Formats a document dictionary into a structured string.

    This function takes a dictionary containing document information (title, major,
    abstract) and formats it into a single string with clear headings.
    
    Args:
        item (dict): A dictionary representing a single document.
    
    Returns:
        str: A formatted string representing the document.
    """
    pos_doc = []
    
    title_keys = ["과제명", "논문 제목"]
    major_keys = ["소속학과"]
    abstract_keys = ["abstract"]
    
    for key in title_keys:
        if key in item:
            pos_doc.append(f"Title: {item[key]}")
            break
    
    for key in major_keys:
        if key in item:
            pos_doc.append(f"Major: {item[key]}")
            break
        
    for key in abstract_keys:
        if key in item:
            pos_doc.append(f"Abstract: {item[key]}")
            break
    
    return "\n".join(pos_doc)


def _load_data_samples(filename):
    """
    Loads and processes data samples from a single JSONL file.

    It iterates through the file, performs a sanity check on each entry,
    preprocesses the document, and extracts all associated queries.

    Args:
        filename (str): The path to the JSONL file.

    Returns:
        list: A list of (queries, positive_document) tuples.
    """
    samples = []
    
    with jsonlines.open(filename) as f:
        for sample in tqdm(f.iter(), desc="Loading data"):
            for _, item in sample.items():
                sanity = _data_sanity_check(item)
                if not sanity:
                    continue
                pos_doc = _preprocess_doc(item)
                queries = []
                for key in item:
                    if key.endswith("쿼리"):
                        queries.extend([v for _, v in item[key].items() if v])
                if len(queries) == 0:
                    break
                samples.append((queries, pos_doc))
    
    return samples

def split_filename(filename):
    """
    Splits a comma-separated string of filenames into a list.

    Args:
        filename (str): A string containing one or more filenames, separated by commas.

    Returns:
        list: A list of individual filenames.
    """
    return filename.split(",")

def load_data_samples(filenames, train=True):
    """
    Loads data samples from multiple files and optionally shuffles them.

    Args:
        filenames (list): A list of file paths to load data from.
        train (bool, optional): If True, the loaded samples are shuffled. Defaults to True.

    Returns:
        list: A list of all loaded samples.
    """
    samples = []
    for filename in filenames:
        samples.extend(_load_data_samples(filename))
        
    if train:
        random.shuffle(samples)

    return samples

def create_hf_dataset(samples):
    """
    Converts a list of samples into a Hugging Face Dataset object.

    Args:
        samples (list): A list of (queries, positive_document) tuples.

    Returns:
        datasets.Dataset: A Hugging Face Dataset object.
    """
    return Dataset.from_list([{"queries": queries, "pos_doc": pos_doc} for queries, pos_doc in samples])

@dataclass
class CustomCollator:
    """
    A custom data collator for SentenceTransformer training.

    This collator is responsible for tokenizing text columns and preparing
    batches for the training process. It dynamically selects one query from the
    list of available queries for each sample at each step. It also handles
    the correct formatting for losses like MultipleNegativesRankingLoss.

    Attributes:
        tokenize_fn (Callable): The tokenizer function from the SentenceTransformer model.
        valid_label_columns (list[str]): A list of valid names for the label column.
    """

    tokenize_fn: Callable
    valid_label_columns: list[str] = field(default_factory=lambda: ["label", "score"])
    _warned_columns: set[tuple[str]] = field(default_factory=set, init=False, repr=False)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collates a list of features into a batch of tensors.

        Args:
            features (list[dict[str, Any]]): A list of dictionary-like features,
                where each feature represents a single training example.

        Returns:
            dict[str, torch.Tensor]: A dictionary of tensors ready for the model.
        """
        column_names = list(features[0].keys())

        # We should always be able to return a loss, label or not:
        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        if tuple(column_names) not in self._warned_columns:
            self.maybe_warn_about_column_order(column_names)

        # Extract the label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        for column_name in column_names:
            # If the prompt length has been set, we should add it to the batch
            if column_name.endswith("_prompt_length") and column_name[: -len("_prompt_length")] in column_names:
                batch[column_name] = torch.tensor([row[column_name] for row in features], dtype=torch.int)
                continue

            tokenized = self.tokenize_fn([random.choice(row[column_name]) if column_name.startswith('queries') else row[column_name] for row in features])
            for key, value in tokenized.items():
                batch[f"{column_name}_{key}"] = value

        return batch

    def maybe_warn_about_column_order(self, column_names: list[str]) -> None:
        """Warn the user if the columns are likely not in the expected order."""
        # A mapping from common column names to the expected index in the dataset
        column_name_to_expected_idx = {
            "anchor": 0,
            "positive": 1,
            "negative": 2,
            "question": 0,
            "answer": 1,
            "query": 0,
            "response": 1,
            "hypothesis": 0,
            "entailment": 1,
            "contradiction": 2,
        }
        for column_name, expected_idx in column_name_to_expected_idx.items():
            if column_name in column_names and column_names.index(column_name) != expected_idx:
                if column_name in ("anchor", "positive", "negative"):
                    proposed_fix_columns = ["anchor", "positive", "negative"]
                elif column_name in ("question", "answer"):
                    proposed_fix_columns = ["question", "answer"]
                elif column_name in ("query", "response"):
                    proposed_fix_columns = ["query", "response"]
                elif column_name in ("hypothesis", "entailment", "contradiction"):
                    proposed_fix_columns = ["hypothesis", "entailment", "contradiction"]

                logger.warning(
                    f"Column {column_name!r} is at index {column_names.index(column_name)}, whereas "
                    f"a column with this name is usually expected at index {expected_idx}. Note that the column "
                    "order can be important for some losses, e.g. MultipleNegativesRankingLoss will always "
                    "consider the first column as the anchor and the second as the positive, regardless of "
                    "the dataset column names. Consider renaming the columns to match the expected order, e.g.:\n"
                    f"dataset = dataset.select_columns({proposed_fix_columns})"
                )
                # We only need to warn once per list of column names to prevent spamming the user
                break

        self._warned_columns.add(tuple(column_names))

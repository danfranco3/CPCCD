import json
import random

def load_multiple_datasets(paths):
    """
    Loads and combines multiple JSON datasets.
    
    Args:
        paths (List[str]): List of JSON file paths.
    
    Returns:
        List[dict]: Combined dataset.
    """
    all_data = []
    for path in paths:
        with open(path, 'r') as f:
            data = json.load(f)
            all_data.extend(data)
    return all_data

def sample_few_shot_examples(dataset, n=1, label=None, seed=42):
    """
    Sample few-shot examples from dataset.
    
    Args:
        dataset (List[dict]): List of code clone samples.
        n (int): Number of examples to sample.
        label (int or None): 1 (positive), 0 (negative), or None for any.
        seed (int): Random seed.
        
    Returns:
        List[dict]: Sampled examples.
    """
    random.seed(seed)
    if label is not None:
        filtered = [ex for ex in dataset if ex['label'] == label]
    else:
        filtered = dataset
    return random.sample(filtered, n)
import argparse
import json
from tokenizers import Tokenizer
from transformers import AutoTokenizer

def extend_tokenizer_and_resize_model(
    tokenizer,
    custom_tokenizer_path: str,
):
    # Load custom tokenizer vocab
    custom_tokenizer = Tokenizer.from_file(custom_tokenizer_path)
    custom_vocab = custom_tokenizer.get_vocab()
    existing_vocab = set(tokenizer.get_vocab().keys())

    # Find new tokens missing from pretrained tokenizer vocab
    new_tokens = [tok for tok in custom_vocab if tok not in existing_vocab]

    # Add new tokens and resize model embeddings
    tokenizer.add_tokens(new_tokens)

    return tokenizer

def print_pair_stats(pairs):
    total = len(pairs)
    positives = sum(1 for p in pairs if p["label"] == 1)
    negatives = total - positives

    print(f"Total samples:       {total}")
    print(f"Positive pairs:      {positives}")
    print(f"Negative pairs:      {negatives}")

    for model_name in ["Salesforce/codet5p-220m", "microsoft/codebert-base"]:
        total_tokens_code1 = 0
        total_tokens_code2 = 0
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = extend_tokenizer_and_resize_model(
            tokenizer,
            custom_tokenizer_path="bpe_cobol_fortran_pascal.json"
        )
        
        for pair in pairs:
            # Tokenize code1 and code2 and count tokens (excluding special tokens if you want)
            tokens_code1 = tokenizer.tokenize(pair["code1"])
            tokens_code2 = tokenizer.tokenize(pair["code2"])

            total_tokens_code1 += len(tokens_code1)
            total_tokens_code2 += len(tokens_code2)

        avg_tokens_code1 = total_tokens_code1 / total if total > 0 else 0
        avg_tokens_code2 = total_tokens_code2 / total if total > 0 else 0

        print(f"Average tokens code1 tokenized by {model_name}: {avg_tokens_code1:.2f}")
        print(f"Average tokens code2 tokenized by {model_name}: {avg_tokens_code2:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)

    print_pair_stats(data)

if __name__ == "__main__":
    main()

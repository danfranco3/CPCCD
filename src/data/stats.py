import argparse
import json

def print_pair_stats(pairs):
    total = len(pairs)
    positives = sum(1 for p in pairs if p["label"] == 1)
    negatives = total - positives

    print(f"Total samples:   {total}")
    print(f"Positive pairs:  {positives}")
    print(f"Negative pairs:  {negatives}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)

    print_pair_stats(data)

if __name__ == "__main__":
    main()

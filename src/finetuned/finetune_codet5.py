import json
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from sklearn.metrics import classification_report, accuracy_score
import os
from code_clone_pkg.data_utils import sample_few_shot_examples, load_multiple_datasets
from code_clone_pkg.dataset import CodeCloneDataset
from code_clone_pkg.prompts import few_shot_prompt
from code_clone_pkg.utils import extend_tokenizer_and_resize_model

# Configuration
MODEL_NAME = "Salesforce/codet5p-220m"
OUTPUT_DIR = "results/finetune"
MAX_LENGTH = 1400
EPOCHS = 4
BATCH_SIZE = 2
CLONE_DATASETS = [
    'python_cobol',
    'java_fortran',
    'js_pascal'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_fp16 = torch.cuda.is_available()

def predict(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
    input_token_count = inputs['input_ids'].shape[-1]
    print(f"Input token count: {input_token_count}")
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def predict_few_shot(example_pairs, target_pair, model, tokenizer):
    prompt = few_shot_prompt(example_pairs, target_pair)
    return predict(prompt, model, tokenizer)

def evaluate_model(model, tokenizer, test_data, raw_examples, output_path):
    model.eval()
    preds = []
    targets = []
    outputs = []

    loader = DataLoader(test_data, batch_size=1)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=10)

            pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip().lower()
            true = tokenizer.decode(labels[0], skip_special_tokens=True).strip().lower()

            pred_label = 1 if 'yes' in pred else 0
            true_label = 1 if 'yes' in true else 0

            preds.append(pred_label)
            targets.append(true_label)

            outputs.append({
                "code1": raw_examples[i]["code1"],
                "code2": raw_examples[i]["code2"],
                "model_output": pred,
                "pred_label": pred_label,
                "true_label": true_label
            })

    print("\nEvaluation Results:")
    print(classification_report(targets, preds, target_names=["Non-clone", "Clone"]))
    print(f"Accuracy: {accuracy_score(targets, preds):.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Detailed results saved to {output_path}")

def eval_two_shot(code_set, support_dataset, test_examples, model, tokenizer):
    targets = []
    preds = []
    outputs = []  # to save detailed info for later

    with torch.no_grad():
        for i, ex in enumerate(test_examples):
            # Sample few-shot support as you do
            few_shot_support = (
                sample_few_shot_examples(support_dataset, n=1, label=1, seed=i) +
                sample_few_shot_examples(support_dataset, n=1, label=0, seed=i + 100)
            )

            prompt = few_shot_prompt(few_shot_support, ex)

            output_text = predict(prompt, model, tokenizer)

            pred_label = 1 if 'yes' in output_text.strip().lower() else 0

            targets.append(ex["label"])
            preds.append(pred_label)

            outputs.append({
                "input": prompt,
                "prediction": output_text,
                "pred_label": pred_label,
                "true_label": ex["label"],
                "code1": ex.get("code1"),
                "code2": ex.get("code2"),
            })

    # After all examples, print metrics
    print("\nEvaluation Results for Few-Shot:")
    print(classification_report(targets, preds, target_names=["Non-clone", "Clone"]))
    print(f"Accuracy: {accuracy_score(targets, preds):.4f}")

    # Save detailed results to JSON file
    output_path = f"{OUTPUT_DIR}/codet5/{code_set}_few_shot.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Detailed results saved to {output_path}")
    
def eval_one_shot(code_set, support_dataset, test_examples, model, tokenizer, neg=False):
    targets = []
    preds = []
    outputs = []  # to save detailed info for later

    with torch.no_grad():
        for i, ex in enumerate(test_examples):
            if neg:
                few_shot_support = sample_few_shot_examples(support_dataset, n=1, label=0, seed=i + 100)
            else:
                few_shot_support = sample_few_shot_examples(support_dataset, n=1, label=1, seed=i)

            prompt = few_shot_prompt(few_shot_support, ex)

            output_text = predict(prompt, model, tokenizer)

            pred_label = 1 if 'yes' in output_text.strip().lower() else 0

            targets.append(ex["label"])
            preds.append(pred_label)

            outputs.append({
                "input": prompt,
                "prediction": output_text,
                "pred_label": pred_label,
                "true_label": ex["label"],
                "code1": ex.get("code1"),
                "code2": ex.get("code2"),
            })

    # After all examples, print metrics
    print(f"\nEvaluation Results for {'Negative' if neg else 'Positive'} One-Shot:")
    print(classification_report(targets, preds, target_names=["Non-clone", "Clone"]))
    print(f"Accuracy: {accuracy_score(targets, preds):.4f}")

    # Save detailed results to JSON file
    output_path = f"{OUTPUT_DIR}/codet5/{code_set}_{'neg' if neg else 'pos'}_one_shot.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Detailed results saved to {output_path}")
    

def run():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    for code_set in CLONE_DATASETS:
        print(f"\n=== Running for dataset: {code_set} ===")
        train_path = "src/data/combined_train.json"
        test_path = f"src/data/rosetta/{code_set}_test.json"

        train_dataset = CodeCloneDataset(train_path, tokenizer, MAX_LENGTH)
        test_dataset = CodeCloneDataset(test_path, tokenizer, MAX_LENGTH)

        train_size = int(len(train_dataset) * 0.8)
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
        if torch.cuda.is_available():
            model.gradient_checkpointing_enable()
            
        tokenizer, model = extend_tokenizer_and_resize_model(
            model,
            tokenizer,
            custom_tokenizer_path="bpe_cobol_fortran_pascal.json"
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")

        training_args = Seq2SeqTrainingArguments(
            output_dir=f"{OUTPUT_DIR}/{code_set}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            eval_accumulation_steps=4,
            predict_with_generate=True,
            logging_dir=f"{OUTPUT_DIR}/{code_set}/logs",
            load_best_model_at_end=True,
            fp16=use_fp16,
            no_cuda=not torch.cuda.is_available(),
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

        trainer.train()
        trainer.save_model(f"{OUTPUT_DIR}/{code_set}/codet5")

        with open(test_path) as f:
            test_examples = json.load(f)

        output_path = f"{OUTPUT_DIR}/codet5/{code_set}_zero_shot.json"
        evaluate_model(model, tokenizer, test_dataset, test_examples, output_path)

        paths = [
            "src/data/codeNet/ruby_go_test.json",
        ]
        
        support_dataset = load_multiple_datasets(paths)
        
        eval_one_shot(code_set, support_dataset, test_examples, model, tokenizer, False)
        eval_one_shot(code_set, support_dataset, test_examples, model, tokenizer, True)

        eval_two_shot(code_set, support_dataset, test_examples, model, tokenizer)
        


if __name__ == "__main__":
    run()

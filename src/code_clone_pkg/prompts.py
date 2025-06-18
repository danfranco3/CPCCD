def few_shot_prompt(example_pairs, target_pair):
    prompt = "Are the following code snippet pairs functionally equivalent?\n\n"
    for ex in example_pairs:
        label = "Yes" if ex["label"] == 1 else "No"
        prompt += f"Code1:\n{ex['code1']}\n\nCode2:\n{ex['code2']}\n\nAnswer: {label}\n\n"
    prompt += f"Code1:\n{target_pair['code1']}\n\nCode2:\n{target_pair['code2']}\n\nAnswer:"
    return prompt

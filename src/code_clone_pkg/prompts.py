def few_shot_prompt(example_pairs, target_pair):
    prompt = "Determine if the following two code snippets functionally equivalent.\n\nAnswer only with Yes or No."
    for ex in example_pairs:
        label = "Yes" if ex["label"] == 1 else "No"
        prompt += f"Example:\n\nCode1:\n{ex['code1']}\n\nCode2:\n{ex['code2']}\n\nAnswer: {label}\n\n"
    if len(example_pairs) != 0:
        prompt += "Now\n\n"
    prompt += f"Code1:\n{target_pair['code1']}\n\nCode2:\n{target_pair['code2']}\n\nAnswer?"
    return prompt

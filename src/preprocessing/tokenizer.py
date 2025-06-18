from tokenizers import Tokenizer, models, trainers
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import json

with open("src/preprocessing/data/cobol_fortran_pascal_code.json", "r") as f:
    code_snippets = json.load(f)

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=30_000,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
)

tokenizer.train_from_iterator(code_snippets, trainer=trainer)

tokenizer.save("bpe_cobol_fortran_pascal.json")

sample_code = "PROGRAM HELLO; WRITE(*,*) 'Hello, World!'; END."
encoded = tokenizer.encode(sample_code, )
print("Tokens:", encoded.tokens)

raw_tokenizer = Tokenizer.from_file("bpe_cobol_fortran_pascal.json")

# Wrap it for Hugging Face Transformers
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<s>",
    sep_token="</s>",
    mask_token="<mask>"
)

# Save in Hugging Face format
wrapped_tokenizer.save_pretrained("tokenizer_ckpt")

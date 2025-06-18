from transformers import PreTrainedTokenizer, PreTrainedModel
from tokenizers import Tokenizer
from typing import Tuple

def extend_tokenizer_and_resize_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    custom_tokenizer_path: str,
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Extend a pretrained tokenizer with tokens from a custom tokenizer,
    resize the model embeddings accordingly, and return updated objects.

    Args:
        model: Pretrained HuggingFace model (e.g., CodeT5).
        tokenizer: Pretrained HuggingFace tokenizer matching the model.
        custom_tokenizer_path: Path to your custom tokenizer JSON file.

    Returns:
        (updated_tokenizer, updated_model)
    """
    # Load custom tokenizer vocab
    custom_tokenizer = Tokenizer.from_file(custom_tokenizer_path)
    custom_vocab = custom_tokenizer.get_vocab()
    existing_vocab = set(tokenizer.get_vocab().keys())

    # Find new tokens missing from pretrained tokenizer vocab
    new_tokens = [tok for tok in custom_vocab if tok not in existing_vocab]

    # Add new tokens and resize model embeddings
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model
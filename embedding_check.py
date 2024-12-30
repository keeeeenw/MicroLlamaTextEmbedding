from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, models

base_model = "path_to_your_microllama_model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("keeeeenw/MicroLlama")

# Create a SentenceTransformer model from the base model and add layers to output embeddings.
base_model = AutoModelForCausalLM.from_pretrained("keeeeenw/MicroLlama")
# base_model = models.Transformer("keeeeenw/MicroLlama", tokenizer_args={'pad_token': '[PAD]'})
# pooling_model = models.Pooling(
#     base_model.get_word_embedding_dimension(),
#     pooling_mode_mean_tokens=True
# )
# model = SentenceTransformer(modules=[base_model, pooling_model])

# Check vocabulary size
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer vocab size after adding padding: {tokenizer.vocab_size}")
print(f"Model embedding matrix size: {base_model.config.vocab_size}")

# # Ensure tokenizer vocab size does not exceed model embedding size
# if tokenizer.vocab_size > model.vocab_size:
#     tokenizer.model_max_length = model.vocab_size
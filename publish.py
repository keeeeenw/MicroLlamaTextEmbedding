from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained("keeeeenw/MicroLlama")
# special_tokens_dict = {'pad_token': '[PAD]'}
# tokenizer.add_special_tokens(special_tokens_dict)

# # Load the model
# base_model = models.Transformer("keeeeenw/MicroLlama", tokenizer_args={'pad_token': '[PAD]'})

# # Check tokenizer and model vocab sizes before resizing
# print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
# print(f"Model vocab size before resize with padding: {base_model.auto_model.config.vocab_size}")

# # Resize model embeddings to match the tokenizer
# base_model.auto_model.resize_token_embeddings(len(tokenizer))

# # Check model vocab size after resizing
# print(f"Model vocab size after resize with padding token: {base_model.auto_model.config.vocab_size}")

# # Pooling layer setup
# pooling_model = models.Pooling(
#     base_model.get_word_embedding_dimension(),
#     pooling_mode_mean_tokens=True
# )

# Construct SentenceTransformer model
model = SentenceTransformer("microllama300m-base-all-nli-stsb-quora-nq-3-epoch")
model.push_to_hub("keeeeenw/MicroLlama-text-embedding")

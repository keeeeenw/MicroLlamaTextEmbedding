import torch
# import os

from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, models
from sentence_transformers.losses import CoSENTLoss, MultipleNegativesRankingLoss, SoftmaxLoss

# 1. Construct the same embedding model for training
# tokenizer = AutoTokenizer.from_pretrained("keeeeenw/MicroLlama")
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Create a SentenceTransformer model from the base model and add layers to output embeddings.
# # base_model = AutoModelForCausalLM.from_pretrained("keeeeenw/MicroLlama")
# base_model = models.Transformer("keeeeenw/MicroLlama", tokenizer_args={'pad_token': '[PAD]'})
# pooling_model = models.Pooling(
#     base_model.get_word_embedding_dimension(),
#     pooling_mode_mean_tokens=True
# )
# model = SentenceTransformer(modules=[base_model, pooling_model])
# print(model)

# model = SentenceTransformer("keeeeenw/MicroLlama", tokenizer_kwargs={'pad_token': '[PAD]'})
# Load tokenizer and add special tokens
tokenizer = AutoTokenizer.from_pretrained("keeeeenw/MicroLlama")
special_tokens_dict = {'pad_token': '[PAD]'}
tokenizer.add_special_tokens(special_tokens_dict)

# Load the model
base_model = models.Transformer("keeeeenw/MicroLlama", tokenizer_args={'pad_token': '[PAD]'})

# Check tokenizer and model vocab sizes before resizing
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Model vocab size before resize: {base_model.auto_model.config.vocab_size}")

# Resize model embeddings to match the tokenizer
base_model.auto_model.resize_token_embeddings(len(tokenizer))

# Check model vocab size after resizing
print(f"Model vocab size after resize: {base_model.auto_model.config.vocab_size}")

# Pooling layer setup
pooling_model = models.Pooling(
    base_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

# Construct SentenceTransformer model
model = SentenceTransformer(modules=[base_model, pooling_model])

# 2. Load several Datasets to train with
# (premise, hypothesis) + label
all_nli_pair_class_train = load_dataset("sentence-transformers/all-nli", "pair-class", split="train[:10000]")

# We can combine all datasets into a dictionary with dataset names to datasets
train_dataset = {
    "all-nli-pair-class": all_nli_pair_class_train,
}

# 3. Load several Datasets to evaluate with
natural_questions_dev = load_dataset("sentence-transformers/natural-questions", split="train[10000:11000]")

# We can use a dictionary for the evaluation dataset too, but we don't have to. We could also just use
# no evaluation dataset, or one dataset.
# eval_dataset = {
#     "natural-questions": natural_questions_dev,
# }

# 4. Load several loss functions to train with
# (sentence_A, sentence_B) + class
softmax_loss = SoftmaxLoss(model, model.get_sentence_embedding_dimension(), 3)

# Create a mapping with dataset names to loss functions, so the trainer knows which loss to apply where.
# Note that you can also just use one loss if all of your training/evaluation datasets use the same loss
losses = {
    "all-nli-pair-class": softmax_loss,
}

# 5. Define a simple trainer, although it's recommended to use one with args & evaluators
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    loss=losses,
    tokenizer=tokenizer,
)
trainer.train()

# 6. save the trained model and optionally push it to the Hugging Face Hub
model.save_pretrained("microllama300m-base-all-nli-stsb-quora-nq")
# model.push_to_hub("microllama300m-base-all-nli-stsb-quora-nq")


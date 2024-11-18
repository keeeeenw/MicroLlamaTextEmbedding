import torch
# import os

from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, models, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CoSENTLoss, MultipleNegativesRankingLoss, SoftmaxLoss

import random
import numpy as np

# 0. Setting seeds
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(42)  # You can choose any seed value

# 1. Model setup
# Setup tokenizer with extra token for padding
tokenizer = AutoTokenizer.from_pretrained("keeeeenw/MicroLlama")
special_tokens_dict = {'pad_token': '[PAD]'}
tokenizer.add_special_tokens(special_tokens_dict)

# Load the model
base_model = models.Transformer("keeeeenw/MicroLlama", tokenizer_args={'pad_token': '[PAD]'})

# Check tokenizer and model vocab sizes before resizing
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Model vocab size before resize with padding: {base_model.auto_model.config.vocab_size}")

# Resize model embeddings to match the tokenizer
base_model.auto_model.resize_token_embeddings(len(tokenizer))

# Check model vocab size after resizing
print(f"Model vocab size after resize with padding token: {base_model.auto_model.config.vocab_size}")

# Pooling layer setup
pooling_model = models.Pooling(
    base_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

# Construct SentenceTransformer model
model = SentenceTransformer(modules=[base_model, pooling_model])

# 2. Load several Datasets to train with
# (anchor, positive)
all_nli_pair_train = load_dataset("sentence-transformers/all-nli", "pair", split="train[:10000]")
# (premise, hypothesis) + label
all_nli_pair_class_train = load_dataset("sentence-transformers/all-nli", "pair-class", split="train[:10000]")
# (sentence1, sentence2) + score
all_nli_pair_score_train = load_dataset("sentence-transformers/all-nli", "pair-score", split="train[:10000]")
# (anchor, positive, negative)
all_nli_triplet_train = load_dataset("sentence-transformers/all-nli", "triplet", split="train[:10000]")
# (sentence1, sentence2) + score
stsb_pair_score_train = load_dataset("sentence-transformers/stsb", split="train[:10000]")
# (anchor, positive)
quora_pair_train = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train[:10000]")
# (query, answer)
natural_questions_train = load_dataset("sentence-transformers/natural-questions", split="train[:10000]")

# We can combine all datasets into a dictionary with dataset names to datasets
train_dataset = {
    "all-nli-pair": all_nli_pair_train,
    "all-nli-pair-class": all_nli_pair_class_train,
    "all-nli-pair-score": all_nli_pair_score_train,
    "all-nli-triplet": all_nli_triplet_train,
    "stsb": stsb_pair_score_train,
    "quora": quora_pair_train,
    "natural-questions": natural_questions_train,
}

# 3. Load several Datasets to evaluate with
# (anchor, positive, negative)
all_nli_triplet_dev = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
# (sentence1, sentence2, score)
stsb_pair_score_dev = load_dataset("sentence-transformers/stsb", split="validation")
# (anchor, positive)
quora_pair_dev = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train[10000:11000]")
# (query, answer)
natural_questions_dev = load_dataset("sentence-transformers/natural-questions", split="train[10000:11000]")

# We can use a dictionary for the evaluation dataset too, but we don't have to. We could also just use
# no evaluation dataset, or one dataset.
eval_dataset = {
    "all-nli-triplet": all_nli_triplet_dev,
    "stsb": stsb_pair_score_dev,
    "quora": quora_pair_dev,
    "natural-questions": natural_questions_dev,
}

# 4. Load several loss functions to train with
# (anchor, positive), (anchor, positive, negative)
mnrl_loss = MultipleNegativesRankingLoss(model)
# (sentence_A, sentence_B) + class
softmax_loss = SoftmaxLoss(model, model.get_sentence_embedding_dimension(), 3)
# (sentence_A, sentence_B) + score
cosent_loss = CoSENTLoss(model)

# Create a mapping with dataset names to loss functions, so the trainer knows which loss to apply where.
# Note that you can also just use one loss if all of your training/evaluation datasets use the same loss
losses = {
    "all-nli-pair": mnrl_loss,
    "all-nli-pair-class": softmax_loss,
    "all-nli-pair-score": cosent_loss,
    "all-nli-triplet": mnrl_loss,
    "stsb": cosent_loss,
    "quora": mnrl_loss,
    "natural-questions": mnrl_loss,
}

# 5. Define a simple trainer, although it's recommended to use one with args & evaluators
train_args = SentenceTransformerTrainingArguments(
    "tmp_trainer-first-10000-3-epoch-retry-1-no-auto-batch",
    per_device_train_batch_size=6,  # Batch size per GPU (or CPU if no GPU is used)
    per_device_eval_batch_size=6,   # Evaluation batch size if you have eval dataset
    # num_train_epochs=10,              # default is 3
    # auto_find_batch_size=True,        # auto adjust batch size
    evaluation_strategy="steps",      # Evaluate every N steps
    eval_steps=500,                  # Perform evaluation every 5000 steps
)
# Default is save for every 500 steps
# https://sbert.net/docs/package_reference/sentence_transformer/training_args.html#sentence_transformers.training_args.SentenceTransformerTrainingArguments.set_save
train_args.set_save(strategy="steps", steps=5000)

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=losses,
    tokenizer=tokenizer,
    args = train_args
)
trainer.train()

# 6. save the trained model and optionally push it to the Hugging Face Hub
model.save_pretrained("microllama300m-base-all-nli-stsb-quora-nq")
# model.push_to_hub("microllama300m-base-all-nli-stsb-quora-nq")


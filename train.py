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
all_nli_pair_train = load_dataset("sentence-transformers/all-nli", "pair", split="train")
# re-name column to allow resume
# Only rename if not resuming
# if 'dataset_name' in all_nli_pair_train.column_names:
#     all_nli_pair_train = all_nli_pair_train.rename_column('dataset_name', 'all_nli_dataset_name')

# Remove the 'dataset_name' column
# if "dataset_name" in all_nli_pair_train.column_names:
#     all_nli_pair_train = all_nli_pair_train.remove_columns("dataset_name")

# (premise, hypothesis) + label
all_nli_pair_class_train = load_dataset("sentence-transformers/all-nli", "pair-class", split="train")
# (sentence1, sentence2) + score
all_nli_pair_score_train = load_dataset("sentence-transformers/all-nli", "pair-score", split="train")
# (anchor, positive, negative)
all_nli_triplet_train = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
# (sentence1, sentence2) + score
stsb_pair_score_train = load_dataset("sentence-transformers/stsb", split="train")
# TODO: 149k row in total, add more training data
# (anchor, positive)
quora_pair_train = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train[:139000]")
# TODO: 100k row in total, add more training data
# (query, answer)
natural_questions_train = load_dataset("sentence-transformers/natural-questions", split="train[:90000]")

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

print("Training sets")
for k,v in train_dataset.items():
    print(k, v.column_names)

# 3. Load several Datasets to evaluate with
# (anchor, positive, negative)
all_nli_triplet_dev = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
# (sentence1, sentence2, score)
stsb_pair_score_dev = load_dataset("sentence-transformers/stsb", split="validation")
# (anchor, positive)
quora_pair_dev = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train[139000:]")
# (query, answer)
natural_questions_dev = load_dataset("sentence-transformers/natural-questions", split="train[90000:]")

# We can use a dictionary for the evaluation dataset too, but we don't have to. We could also just use
# no evaluation dataset, or one dataset.
eval_dataset = {
    "all-nli-triplet": all_nli_triplet_dev,
    "stsb": stsb_pair_score_dev,
    "quora": quora_pair_dev,
    "natural-questions": natural_questions_dev,
}

print("Eval sets")
for k,v in eval_dataset.items():
    print(k, v.column_names)

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

# TODO: this does not work because warmup step will actually increase the LR
# Maybe start with a lower learning rate that's ~ 10 epoch
# Or our LR should decrease at constant rate, so that 10 epoch should have a lower LR in the end

# Config learning rate based on batch size, training set size, and num epoch
# train_batch_size = 6  # Batch size
# eval_batch_size = 6  # Batch size
# num_epochs = 10  # Number of epochs
# warmup_ratio = 0.1  # Warm up for 10% of the training steps
# training_data_size = sum([len(d) for d in train_dataset.values()])  # Updated size of your training data

# # Recalculate the number of training steps
# # This should be close to 32883 for train_batch_size 6, num_epochs 3 for the initial model with limited training data.
# num_training_steps = (training_data_size // train_batch_size) * num_epochs

# # Recalculate warmup steps (e.g., 10% of total training steps)
# warmup_steps = int(warmup_ratio * num_training_steps)

# # 5. Define a simple trainer, although it's recommended to use one with args & evaluators
# train_args = SentenceTransformerTrainingArguments(
#     "tmp_trainer-first-10000-10-epoch-retry-3-cosine-custom-steps",
#     per_device_train_batch_size=train_batch_size,  # Batch size per GPU (or CPU if no GPU is used)
#     per_device_eval_batch_size=eval_batch_size,   # Evaluation batch size if you have eval dataset
#     num_train_epochs=num_epochs,  # Number of epochs
#     learning_rate=5e-5,            # always start with the same default learning rate.
#     # max_steps=num_training_steps,  # Fixed training steps for consistent decay
#     # warmup_steps=warmup_steps,
#     # weight_decay=0.01,
#     # auto_find_batch_size=True,        # auto adjust batch size
#     evaluation_strategy="steps",      # Evaluate every N steps
#     eval_steps=500,                  # Perform evaluation every 5000 steps
# )
# # Default is save for every 500 steps
# # https://sbert.net/docs/package_reference/sentence_transformer/training_args.html#sentence_transformers.training_args.SentenceTransformerTrainingArguments.set_save
# train_args.set_save(strategy="steps", steps=5000)

# # This will not work because it was auto adjusted.
# # train_args.set_lr_scheduler(num_epochs=num_epochs)
# train_args.set_lr_scheduler(num_epochs=num_epochs,
#                             name="cosine",
#                             max_steps=num_training_steps,
#                             warmup_ratio=warmup_ratio,
#                             warmup_steps=warmup_steps)

train_batch_size = 4  # Batch size
eval_batch_size = 4  # Batch size
num_epochs = 3  # Number of epochs
train_args = SentenceTransformerTrainingArguments(
    "tmp_trainer-full-3-epoch-linear-lr-1e-5-batch-4",
    per_device_train_batch_size=train_batch_size,  # Batch size per GPU (or CPU if no GPU is used)
    per_device_eval_batch_size=eval_batch_size,   # Evaluation batch size if you have eval dataset
    num_train_epochs=num_epochs,  # Number of epochs
    learning_rate=1e-5,            # converages too quickly?
    # learning_rate=3e-5,            # converages too quickly?
    # learning_rate=2e-5,            # always start with the same default learning rate.
    # max_steps=num_training_steps,  # Fixed training steps for consistent decay
    # warmup_steps=warmup_steps,
    # weight_decay=0.01,
    auto_find_batch_size=True,        # auto adjust batch size
    evaluation_strategy="steps",      # Evaluate every N steps
    eval_steps=2500,                  # Perform evaluation every 5000 steps
    save_total_limit=20,
    bf16=True,
)
# Default is save for every 500 steps
# https://sbert.net/docs/package_reference/sentence_transformer/training_args.html#sentence_transformers.training_args.SentenceTransformerTrainingArguments.set_save
train_args.set_save(strategy="steps", steps=5000)

# TODO: for some reason it restarted training around
"""
{'loss': 0.7495, 'grad_norm': 0.0019107084954157472, 'learning_rate': 9.433869240733956e-06, 'epoch': 0.17}
  6%|█████                                                                                    | 127421/2243298 [6:34:58<109:18:46,  5.38it/s]
{'loss': 0.7143, 'grad_norm': 0.016145436093211174, 'learning_rate': 9.997771138743048e-06, 'epoch': 0.0}        | 0/2243298 [00:00<?, ?it/s]
{'loss': 0.6057, 'grad_norm': 0.34214434027671814, 'learning_rate': 9.995542277486094e-06, 'epoch': 0.0}
"""
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=losses,
    tokenizer=tokenizer,
    args = train_args
)
# trainer.train(resume_from_checkpoint=True)
trainer.train()

# 6. save the trained model and optionally push it to the Hugging Face Hub
model.save_pretrained("microllama300m-base-all-nli-stsb-quora-nq")
# model.push_to_hub("microllama300m-base-all-nli-stsb-quora-nq")


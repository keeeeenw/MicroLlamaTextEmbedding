# See https://huggingface.co/collections/tomaarsen/training-with-prompts-672ce423c85b4d39aed52853 for some already trained models

import logging
import random
import os

import numpy
import torch
from datasets import Dataset, load_dataset
from datetime import datetime

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    models
)
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from transformers import AutoTokenizer

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
random.seed(12)
torch.manual_seed(12)
numpy.random.seed(12)

# Feel free to adjust these variables:
use_prompts = True
include_prompts_in_pooling = True

# 1. Load a model to finetune with 2. (Optional) model card data
model_name = "keeeeenw/MicroLlama"
# model = SentenceTransformer(
#     "microsoft/mpnet-base",
#     model_card_data=SentenceTransformerModelCardData(
#         language="en",
#         license="apache-2.0",
#         model_name="MPNet base trained on Natural Questions pairs",
#     ),
# )

# 1. Model setup
# Setup tokenizer with extra token for padding
tokenizer = AutoTokenizer.from_pretrained(model_name)
special_tokens_dict = {'pad_token': '[PAD]'}
tokenizer.add_special_tokens(special_tokens_dict)

# Load the model
base_model = models.Transformer(model_name, tokenizer_args={'pad_token': '[PAD]'})

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

# Special setup for prompt model
model.set_pooling_include_prompt(include_prompts_in_pooling)

# 2. (Optional) Define prompts
if use_prompts:
    query_prompt = "query: "
    corpus_prompt = "document: "
    prompts = {
        "query": query_prompt,
        "answer": corpus_prompt,
    }

# 3. Load a dataset to finetune on
dataset = load_dataset("sentence-transformers/natural-questions", split="train")

# remove rows with long content to avoid OOM
def filter_long_anchors(example):
    for c in example.keys():
        # most data has 1.36k length
        if isinstance(example[c], str) and len(example[c]) > 1370:
            return False
    return True

print("Data before filtering")
print(dataset)
dataset = dataset.filter(filter_long_anchors)
print("Data after filtering")
print(dataset)

dataset_dict = dataset.train_test_split(test_size=1_000, seed=12)
train_dataset: Dataset = dataset_dict["train"]
eval_dataset: Dataset = dataset_dict["test"]

# 4. Define a loss function
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)

# 5. (Optional) Specify training arguments
run_name = f"np-{model_name}"
if use_prompts:
    run_name += "-prompts"
if not include_prompts_in_pooling:
    run_name += "-exclude-pooling-prompts"
train_batch_size = 256
eval_batch_size = 256
num_epochs = 20
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=5,
    logging_steps=10,
    logging_first_step=True,
    run_name=f"{run_name}-{train_batch_size}-{num_epochs}",  # Will be used in W&B if `wandb` is installed
    seed=12,
    prompts=prompts if use_prompts else None,
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = NanoBEIREvaluator(
    query_prompts=query_prompt if use_prompts else None,
    corpus_prompts=corpus_prompt if use_prompts else None,
    batch_size=8 # override batch size for OOM
)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 8. Save the trained model
model.save_pretrained(f"models/{run_name}/final")

# 9. (Optional) Create an evaluator & evaluate the base model
output_path = os.path.join("sentence_transformer_training_dev_results", run_name + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
if not os.path.exists(output_path):
    os.mkdir(output_path)
results = dev_evaluator(model, output_path=output_path)
print("---------------- Evaluations results starts -----------------")
print(results)
print("---------------- Evaluations results ends -------------------")

# 9. (Optional) Push it to the Hugging Face Hub
# model.push_to_hub(run_name)

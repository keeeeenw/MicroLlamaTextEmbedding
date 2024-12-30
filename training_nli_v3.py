"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with GISTEmbedLoss, using all-MiniLM-L6-v2 as an efficient guiding model. Entailments are positive pairs and the contradiction
on AllNLI dataset is added as a hard negative. At every 10% training steps, the model is evaluated on the STS benchmark dataset

Usage:
python training_nli_v3.py

OR
python training_nli_v3.py pretrained_transformer_model_name
"""

import logging
import sys
import traceback
import torch
import random
import numpy as np

from datetime import datetime
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments
from transformers import AutoTokenizer

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = sys.argv[1] if len(sys.argv) > 1 else "keeeeenw/MicroLlama"
# TODO: retry with 32 by loading the existing checkpoint and skip the large batch dataset
train_batch_size = 32  # The larger you select this, the better the results (usually). But it requires more GPU memory
# max_seq_length = 75
num_epochs = 3

# Save path of the model
output_dir = (
    "output/training_nli_v3_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# 0. Setting seeds
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(42)  # You can choose any seed value

# # 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
# # create one with "mean" pooling.
# model = SentenceTransformer(model_name)

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

# 2. Load the AllNLI dataset: https://huggingface.co/datasets/sentence-transformers/all-nli
# We'll start with 10k training samples, but you can increase this to get a stronger model
logging.info("Read AllNLI train dataset")
# train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train").select(range(10000))
# eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev").select(range(1000))
# Use full dataset
train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
logging.info("Training data before filtering")
logging.info(train_dataset)
logging.info("Eval data before filtering")
logging.info(eval_dataset)

# 2.5 Filter training data to remove the dataset that is too long which would cause OOM
def filter_long_anchors(example):
    return len(example["anchor"]) <= 1600
train_dataset = train_dataset.filter(filter_long_anchors)
logging.info("Training data after filtering") # should remove two rows
logging.info(train_dataset)

# 3. Define our training loss: https://sbert.net/docs/package_reference/sentence_transformer/losses.html#gistembedloss
# The guiding model
guide_model = SentenceTransformer("all-MiniLM-L6-v2")
train_loss = losses.GISTEmbedLoss(model, guide_model)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=stsb_eval_dataset["sentence1"],
    sentences2=stsb_eval_dataset["sentence2"],
    scores=stsb_eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)
logging.info("Evaluation before training:")
dev_evaluator(model)

# 5. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=20,
    logging_steps=100,
    run_name=f"nli-v3-microllama-{train_batch_size}-{num_epochs}-data-filter-1600",  # Will be used in W&B if `wandb` is installed
    # auto_find_batch_size=True, # auto adjust batch size
)

# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 7. Evaluate the model performance on the STS Benchmark test dataset
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["sentence1"],
    sentences2=test_dataset["sentence2"],
    scores=test_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-test",
)
test_evaluator(model)

# 8. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)

# # 9. (Optional) save the model to the Hugging Face Hub!
# # It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
# model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
# try:
#     model.push_to_hub(f"{model_name}-nli-v3")
# except Exception:
#     logging.error(
#         f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
#         f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
#         f"and saving it using `model.push_to_hub('{model_name}-nli-v3')`."
#     )

import torch
# import os

from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, models, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CoSENTLoss, MultipleNegativesRankingLoss, SoftmaxLoss


# 0. Load a model to train
# https://stackoverflow.com/questions/76051807/automodelforcausallm-for-extracting-text-embeddings
# tokenizer = AutoTokenizer.from_pretrained("keeeeenw/MicroLlama")
# base_model = AutoModelForCausalLM.from_pretrained("keeeeenw/MicroLlama")

# texts = [
#     "this is a test",
#     "this is another test case with a different length",
# ] 
# tokenizer.pad_token = tokenizer.eos_token
# t_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


# with torch.no_grad():
#     last_hidden_state = base_model(**t_input, output_hidden_states=True).hidden_states[-1]


# weights_for_non_padding = t_input.attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(0)

# sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
# num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
# sentence_embeddings = sum_embeddings / num_of_none_padding_tokens
# print(sentence_embeddings)

# 1. Construct the same embedding model for training
# tokenizer = AutoTokenizer.from_pretrained("keeeeenw/MicroLlama")
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# # Create a SentenceTransformer model from the base model and add layers to output embeddings.
# # # base_model = AutoModelForCausalLM.from_pretrained("keeeeenw/MicroLlama")
# # base_model = models.Transformer("keeeeenw/MicroLlama", tokenizer_args={'pad_token': '[PAD]'})
# # pooling_model = models.Pooling(
# #     base_model.get_word_embedding_dimension(),
# #     pooling_mode_mean_tokens=True
# # )
# # model = SentenceTransformer(modules=[base_model, pooling_model])
# # print(model)

# model = SentenceTransformer("keeeeenw/MicroLlama", tokenizer_kwargs={'pad_token': '[PAD]'})

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
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=losses,
    tokenizer=tokenizer,
    args = SentenceTransformerTrainingArguments(
        "tmp_trainer",
        per_device_train_batch_size=6,  # Batch size per GPU (or CPU if no GPU is used)
        per_device_eval_batch_size=6,   # Evaluation batch size if you have eval dataset
    )
)
trainer.train()

# 6. save the trained model and optionally push it to the Hugging Face Hub
model.save_pretrained("microllama300m-base-all-nli-stsb-quora-nq")
# model.push_to_hub("microllama300m-base-all-nli-stsb-quora-nq")


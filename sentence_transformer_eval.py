import os

from sentence_transformers import (
    SentenceTransformer,
    models
)
from sentence_transformers.evaluation import NanoBEIREvaluator
from transformers import AutoTokenizer

use_prompts = True
include_prompts_in_pooling = True

def create_model(model_name, use_prompts=False, include_prompts_in_pooling=False):
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
    if include_prompts_in_pooling:
        print("Setting up prompts in pooling")
        model.set_pooling_include_prompt(include_prompts_in_pooling)

    # 2. (Optional) Define prompts
    if use_prompts:
        print("Setting up prompts for the model")
        query_prompt = "query: "
        corpus_prompt = "document: "
        prompts = {
            "query": query_prompt,
            "answer": corpus_prompt,
        }
        return model, prompts
    
    return model, {}

eval_models = {
    "models/np-keeeeenw/MicroLlama-prompts/final": {
        "use_prompts": True,
        "include_prompts_in_pooling": True
    },
    "output/training_nli_v3_keeeeenw-MicroLlama-2024-12-28_11-53-11/final": {
        "use_prompts": False,
        "include_prompts_in_pooling": False
    },
    "keeeeenw/MicroLlama-text-embedding": {
        "use_prompts": False,
        "include_prompts_in_pooling": False
    },    
    "tmp_trainer_batch_6_epoch_3_released_v1/checkpoint-32500": {
        "use_prompts": False,
        "include_prompts_in_pooling": False
    },    
}

for model_name, args in eval_models.items():
    print("Evaluating", model_name, "Params:", args)
    model, prompts = create_model(model_name, **args)
    dev_evaluator = NanoBEIREvaluator(
        query_prompts=prompts.get("query", None),
        corpus_prompts=prompts.get("answer", None),
        batch_size=8,
        write_csv=True,
        show_progress_bar=True
    )
    model_name_formatted = model_name.replace("/", "-")
    output_path = os.path.join("sentence_transformer_eval_results", model_name_formatted)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    results = dev_evaluator(model, output_path=output_path)
    print("---------------- Evaluations results starts -----------------")
    print(results)
    print("---------------- Evaluations results ends -------------------")


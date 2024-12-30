# based on https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_model.md

import mteb

# load a model from the hub (or for a custom implementation see https://github.com/embeddings-benchmark/mteb/blob/main/docs/reproducible_workflow.md)
# Best model
# model = mteb.get_model("keeeeenw/MicroLlama-text-embedding")
# Not good
# model = mteb.get_model("microllama-vast-ai-round-1-70000-gpu-4-batch-8")
# Not good
# model = mteb.get_model("tmp_trainer-first-10000-10-epoch/checkpoint-105000")
# Not good
# model = mteb.get_model("tmp_trainer-first-10000-10-epoch/checkpoint-30000")
# Best model another checkpoint
# model = mteb.get_model("tmp_trainer_batch_6_epoch_3_released_v1/checkpoint-30000")

# model = mteb.get_model("output/training_nli_v3_keeeeenw-MicroLlama-2024-12-27_21-09-31/final")

model = mteb.get_model("output/training_nli_v3_keeeeenw-MicroLlama-2024-12-28_11-53-11/final")

# tasks = mteb.get_tasks(...) # get specific tasks
# # or 
# https://github.com/embeddings-benchmark/mteb/blob/fccf034bd78d74917f9d8fb6053e473fb03e86d8/mteb/benchmarks/benchmarks.py#L71
# English benchmarks from MTEB
# Classic benchmark (eng, classic) has the depreciated ArxivClusteringS2S which would lead to network disconnection 
tasks = mteb.get_benchmark("MTEB(eng, beta)")

evaluation = mteb.MTEB(tasks=tasks)
# the default 64 will get OOM
evaluation.run(model, output_folder="results", encode_kwargs={'batch_size':16})

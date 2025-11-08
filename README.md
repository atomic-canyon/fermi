# Fermi Tools

This repository provides an easy-to-use CLI for benchmarking sparse embedding models on standard IR benchmarks, specifically [BEIR](https://github.com/beir-cellar/beir). Built on top of the [BEIR framework](https://github.com/beir-cellar/beir) and [Milvus](https://milvus.io/) for high-performance vector search, this tool allows users to evaluate sparse embedding models against various datasets, including [FermiBench](https://huggingface.co/datasets/atomic-canyon/FermiBench), a nuclear-specific information retrieval benchmark. With this release, we aim to fill a gap in the community by offering an open, standardized method to assess sparse embeddings models.

## Prerequisites

### Milvus Setup

Start Milvus using Docker Compose:

```bash
docker-compose up -d
```

This will start:
- **Milvus standalone**: Vector database on port 19530
- **MinIO**: Object storage for Milvus
- **etcd**: Metadata storage

Wait for all services to be healthy before running benchmarks.

## Run a Benchmark for a Sparse Embedding Model

Build docker
```bash
docker build -t fermi:latest .
```

Run fermi benchmark
```bash
docker run --rm -it \
    --gpus all --network milvus \
    -v "$(pwd)/src/beir_cache:/app/src/beir_cache" \
    -e MILVUS_HOST=milvus-standalone \
    fermim:latest python ./src/run_beir.py --dataset fermi --model atomic-canyon/fermi-512
```

Run scifact benchmark
```bash
docker run --rm -it \
    --gpus all --network milvus \
    -v "$(pwd)/src/beir_cache:/app/src/beir_cache" \
    -e MILVUS_HOST=milvus-standalone \
    fermim:latest python ./src/run_beir.py --dataset scifact --model atomic-canyon/fermi-512
```
See [BEIR Github](https://github.com/beir-cellar/beir) for list of all benchmarks



### Expected Results

example command to run agianst Milvus
```bash
docker run --rm -it \
    --gpus all --network milvus \
    -v "$(pwd)/src/beir_cache:/app/src/beir_cache" \
    -e MILVUS_HOST=milvus-standalone -e HF_TOKEN=YOUR_TOKEN_WITH_FERMI_ACCESS \
    fermi:latest python ./src/run_beir.py --dataset fermi --model atomic-canyon/fermi-512

```

**"NDCG@10": 0.73883** like indicated on fermi-512 model card

```json
{"ndcg": {"NDCG@1": 0.58974, "NDCG@3": 0.69479, "NDCG@5": 0.71909, "NDCG@10": 0.73883, "NDCG@100": 0.76223, "NDCG@1000": 0.76406}, "map": {"MAP@1": 0.55556, "MAP@3": 0.6609, "MAP@5": 0.67545, "MAP@10": 0.6835, "MAP@100": 0.68982, "MAP@1000": 0.68993}, "recall": {"Recall@1": 0.55556, "Recall@3": 0.77265, "Recall@5": 0.8281, "Recall@10": 0.88793, "Recall@100": 0.98718, "Recall@1000": 1.0}, "precision": {"P@1": 0.58974, "P@3": 0.27778, "P@5": 0.18077, "P@10": 0.09679, "P@100": 0.01128, "P@1000": 0.00114}, "mrr": {"MRR@1": 0.58974, "MRR@3": 0.68162, "MRR@5": 0.69573, "MRR@10": 0.70449, "MRR@100": 0.70686, "MRR@1000": 0.70696}, "hr": {"HR@1": 0.58974, "HR@3": 0.79487, "HR@5": 0.85256, "HR@10": 0.91667, "HR@100": 0.98718, "HR@1000": 1.0}, "args": {"dataset": "atomic-canyon/FermiBench", "split": "test", "cache_dir": "/app/src/beir_cache", "note": null, "jsonl_log": "/app/src/beir_cache/beir_log.jsonl", "model": "atomic-canyon/fermi-512", "tokenizer": "atomic-canyon/fermi-512", "rerank_model": null, "sparse_index_dir": null, "no_chunking": false, "no_flops": false, "max_tokens": 512, "token_overlap": 48, "paragraph_separator": "\n\n"}, "eval_time": "2025-11-08T02:03:02.765557", "flops": 0.0, "avg_doc_sparsity": 239.41350000002194, "avg_query_sparsity": 48.224358974357756, "total_num_embeddings": 141876}
```

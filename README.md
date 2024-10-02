# Fermi Tools

This repository provides an easy-to-use CLI for benchmarking sparse embedding models on standard IR benchmarks, specifically [BEIR](https://github.com/beir-cellar/beir). Built on top of the [BEIR framework](https://github.com/beir-cellar/beir) and [Pyserini](https://github.com/castorini/pyserini), this tool allows users to evaluate sparse embedding models against various datasets, including [FermiBench](https://huggingface.co/datasets/atomic-canyon/FermiBench), a nuclear-specific information retrieval benchmark. With this release, we aim to fill a gap in the community by offering an open, standardized method to assess sparse embeddings models.

## Run a Benchmark for a Sparse Embedding Model

Build docker
```bash
docker build -t fermi:latest .
```

Run fermi benchmark
```bash
docker run --rm -it \
    --gpus all -v "$(pwd)/src/beir_cache:/app/src/beir_cache" \
    fermi:latest python ./src/run_beir.py --dataset fermi --model atomic-canyon/fermi-512
```

Run scifact benchmark
```bash
docker run --rm -it \
    --gpus all -v "$(pwd)/src/beir_cache:/app/src/beir_cache" \
    fermi:latest python ./src/run_beir.py --dataset scifact --model atomic-canyon/fermi-512
```
See [BEIR Github](https://github.com/beir-cellar/beir) for list of all benchmarks

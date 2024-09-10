# fermi
Tools for Fermi



# Run a benchmark on sparse embedding model

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
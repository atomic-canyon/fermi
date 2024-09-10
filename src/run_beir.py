from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search import BaseSearch
from datasets import Dataset
from datetime import datetime
from decouple import config
from huggingface_hub import snapshot_download
from llama_index.core.text_splitter import SentenceSplitter
from pyserini.search.lucene import LuceneImpactSearcher
from sentence_transformers import CrossEncoder
from sparse_embeddings import create_sparse_index, output_to_weight_dicts, get_encoded_query_token_weight_dicts, lookup_sparse_collection_text
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import Dict, List, Tuple
from typing import List
import argparse
import collections
import json
import logging
import numpy as np
import os
import torch

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_BATCH_SIZE = config('GPU_BATCH_SIZE', default=16, cast=int)
NUM_PROC = config('NUM_PROC', default=16, cast=int)

# hitrate @ given k value
# what fraction of queries resulted in a responsive document being returned in the first k?
def hr_k(
        qrels: Dict[str, Dict[str, int]], #question_id to a list of doc_ids with relevance scores
        results: Dict[str, Dict[str, float]], #question_id to a list of doc_ids with relevance scores
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    hr_k = {}
    for k in k_values:
        num_questions = 0
        num_correct = 0
        for query_id in qrels.keys():
            if not qrels[query_id]: continue #question has no known relevant docs. skip it
            valid_docs = set(k for k,v in qrels[query_id].items() if v > 0)
            if not len(valid_docs): continue #question has no known relevant docs. skip it
            num_questions += 1

            res_docs = list(results[query_id].items())
            res_docs.sort(key=lambda x: x[1], reverse=True)
            res_docs = res_docs[:k]
            res_docs = set(k for k,v in res_docs)
            if res_docs & valid_docs:
                num_correct += 1

        hr_k[f"HR@{k}"] = round(num_correct/num_questions, 5)
    return hr_k

class UnpickleableLoggingFilter(logging.Filter):
    def filter(self, rec):
        return "unpickleable private attribute" not in rec.getMessage()

def corpus_to_dataset(corpus):
    def doc_generator():
        for id, doc in corpus.items():
            doc['docKey'] = id
            yield doc
    return Dataset.from_generator(doc_generator)

def flatten_chunks(ds):
    # convert input dataset from a list of lists of texts to just a list of texts
    def _flatten_chunks(batch):
        ret = {'text': []}
        ret['docKey'] = []
        for idx in range(len(batch['docKey'])):
            for chunk in batch['text'][idx]:
                ret['docKey'].append(batch['docKey'][idx])
                ret['text'].append(chunk)
        return ret
    ds = ds.filter(lambda e: bool(e['text']), num_proc=NUM_PROC, desc='remove empty chunks (prob not needed)')
    ds = ds.map(_flatten_chunks, batched=True, remove_columns=ds.column_names, batch_size=1, num_proc=NUM_PROC, desc='flattening chunks')
    return ds

def chunk_dataset_llamaindex_sentence(ds, tokenizer_model="bert-base-uncased", max_tokens=512, token_overlap=64):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    splitter = SentenceSplitter(
        chunk_size=max_tokens,
        chunk_overlap=token_overlap,
        tokenizer=tokenizer.tokenize,
        paragraph_separator="\n\n"
    )
    def chunkit(example, **kw):
        return dict(text=splitter.split_text(example['text']))
    # pass in kwargs to invalidate huggingface datasets cache
    return ds.map(
        chunkit,
        num_proc=NUM_PROC,
        desc='chunking docs',
        fn_kwargs=dict(tokenizer_model=tokenizer_model, max_tokens=max_tokens, token_overlap=token_overlap),
    )

_special_token_ids = {} #cache special token ids
def _get_sparse_vector(tokenizer, feature, output):
    #get special token ids
    tokenizer_name = tokenizer if isinstance(tokenizer, str) else tokenizer.name_or_path
    st_ids = _special_token_ids.get(tokenizer_name)
    if st_ids is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        st_ids = [tokenizer.vocab[token] for token in tokenizer.special_tokens_map.values() if type(token) == str]
        _special_token_ids[tokenizer_name] = st_ids
    #get sparse vector in dense format
    values, _ = torch.max(output*feature["attention_mask"].unsqueeze(-1), dim=1)
    values = torch.log(1 + torch.relu(values))
    values[:,st_ids] = 0 #zero out special tokens
    return values

def sparse_embed(model, tokenizer, texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs.to(device)
    with torch.no_grad():
        if 'token_type_ids' in inputs: del inputs['token_type_ids']
        outputs = model(**inputs)
        embeddings = _get_sparse_vector(tokenizer, inputs, outputs['logits'])
        embeddings = embeddings.cpu().detach()
    return [r.numpy() for r in embeddings]

def sparse_embed_dataset(ds, model, tokenizer):
    return ds.map(lambda batch: {'embeddings': sparse_embed(model, tokenizer, batch['text'])},
                  batched=True, batch_size=GPU_BATCH_SIZE, num_proc=1, desc=f'computing sparse embeddings')

class SparseSearchAdapter(BaseSearch):
    def __init__(self, model, tokenizer, rerank_model=None, sparse_index_dir=None, compute_flops=True, do_chunking=True, max_tokens=512):
        self.model = model
        self.tokenizer = tokenizer
        self.rerank_model = rerank_model
        self.sparse_index_dir = sparse_index_dir
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
        self.do_chunking = do_chunking
        self.compute_flops = compute_flops
        self.max_tokens = max_tokens
        self.doc_sparsity_dist = collections.Counter() # used to store sparsity stats and compute flops
        self.query_sparsity_dist = collections.Counter() # used to store sparsity stats and compute flops
        self.flops = None #if compute_flops is set, this will be populated
        self.avg_query_sparsity = None
        self.avg_doc_sparsity = None
        self.results = {}

    def search(self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int, *args, **kwargs) -> Dict[str, Dict[str, float]]:

        model = AutoModelForMaskedLM.from_pretrained(self.model)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        reverse_voc = self.reverse_voc

        logger.info("Converting corpus to dataset...")
        doc_ds = corpus_to_dataset(corpus)
        if self.do_chunking:
            logger.info("Chunking corpus dataset...")
            # token_overlap is a magic number, chosen without much thought, but must be consistent
            doc_ds = chunk_dataset_llamaindex_sentence(
                doc_ds, tokenizer_model=self.tokenizer,
                max_tokens=self.max_tokens, token_overlap=64
            )
            doc_ds = flatten_chunks(doc_ds)

        logger.info("Starting embedding compute...")
        model.to(device)
        model.eval()
        doc_ds = sparse_embed_dataset(doc_ds, model, tokenizer)
        logger.info("Embedding compute complete.")

        if self.compute_flops:
            def _compute_stats(ex):
                vec = np.array(ex['embeddings'])
                vec = output_to_weight_dicts([vec], reverse_voc)[0]
                for k, v in vec.items():
                    self.doc_sparsity_dist[k] += 1
                return
            sample_size = min(doc_ds.num_rows, 10_000)
            doc_ds_sample = doc_ds.shuffle().select(range(sample_size))
            doc_ds_sample.filter(_compute_stats, num_proc=1, desc='compute sparsity stats')

        lucene_index_path = create_sparse_index(doc_ds, self.tokenizer, text_key='text', save_text=bool(self.rerank_model), save_dir=self.sparse_index_dir)

        rerank_model = None
        if self.rerank_model:
            logger.info("Rerank model is set. Loading rerank model '{self.rerank_model}'")
            rerank_model = CrossEncoder(self.rerank_model)
            rerank_model.model.to(device)
            rerank_model.model.eval()

        query_sparsity_dist = self.query_sparsity_dist if self.compute_flops else None
        class QueryEncoder():
            def encode(self, texts, **kwargs):
                ret = sparse_embed(model, tokenizer, texts)
                ret = output_to_weight_dicts(ret, reverse_voc)
                if query_sparsity_dist is not None: #compute sparsity stats if needed
                    for vec in ret:
                        for k, v in vec.items():
                            query_sparsity_dist[k] += 1
                ret = get_encoded_query_token_weight_dicts(ret)[0]
                return ret

        searcher = LuceneImpactSearcher(lucene_index_path, QueryEncoder())
        for qid, query in tqdm(queries.items(), desc='searching queries'):
            # this section increases k to implement score-max https://arxiv.org/pdf/2305.18494
            # by default bier's EvaluateRetrieval.retrieve() passes top_k = 1000
            hits = searcher.search(query, k=top_k*2)
            if rerank_model:
                scores = rerank_model.predict([(query, lookup_sparse_collection_text(lucene_index_path, hit.docid)) for hit in hits], batch_size=GPU_BATCH_SIZE, show_progress_bar=False)
                for hit, score in zip(hits, scores):
                    hit.score = score #overwrite the score with the rerank score

            # sort the hits with LOWEST score first
            hits.sort(key=lambda x: x.score)

            # dedup the hits, overwriting the lower scores with later, higher scores
            # this will only come into play if there are multiple chunks for the same document
            res = {hit.docid.rsplit("_", 1)[0]: float(hit.score) for hit in hits}
            res = list(res.items())

            # reorder the hits with highest scores first
            res.sort(key=lambda x: x[1], reverse=True)
            # actually respect the real top_k that was passed in now that we've deduped
            res = dict(res[:top_k])

            # store the results on the class instance because that's what BIER does?
            self.results[qid] = res

        # Free up GPU memory & cleanup
        model.to('cpu')
        if rerank_model: rerank_model.model.to('cpu')
        torch.cuda.empty_cache()

        if self.compute_flops:
            self.flops = 0
            self.doc_sparsity_dist = {k:v/sample_size for k,v in self.doc_sparsity_dist.items()} #norm to sample size
            self.avg_doc_sparsity = sum(self.doc_sparsity_dist.values())
            self.query_sparsity_dist = {k:v/len(queries) for k,v in self.query_sparsity_dist.items()} #norm to sample size
            self.avg_query_sparsity = sum(self.query_sparsity_dist.values())
            for k, v in self.doc_sparsity_dist.items():
                self.flops += v * self.query_sparsity_dist.get(k, 0)
            logger.info(f"Computed FLOPS: {self.flops}")

        return self.results

def main():
    parser = argparse.ArgumentParser(description="CLI for running IR benchmarks on various datasets using BEIR. Results are written as jsonl to stdout.")
    parser.add_argument('--dataset', type=str, required=True, help='Determines what benchmark to run. In additiona to the BEIR datasets and fermi, you can also pass in a path to a dataset directory or a URL to a zip file.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'dev', 'test'], help='Specifies the dataset split to use. Valid options are: train, dev, test. Default is "test".')
    parser.add_argument('--cache-dir', type=str, default='./beir_cache', help='Directory where BEIR data is downloaded and cached', metavar='PATH')
    parser.add_argument('--note', type=str, help='A free form string to describe the setup being evaluated.')
    parser.add_argument('--jsonl-log', type=str, default='./beir_cache/beir_log.jsonl', help='File path where results are appended in JSONL format. Results are also written to stdout', metavar='PATH')
    parser.add_argument('--model', type=str, required=True, help='Name of the model to use for sparse search. Must be a valid model name or path.')
    parser.add_argument('--tokenizer', type=str, help='Name of the tokenizer to use for the model. Default is --model if not specified')
    parser.add_argument('--rerank-model', type=str, help='Name of the model to use for reranking. Must be a valid model name or path.')
    parser.add_argument('--sparse-index-dir', type=str, help='directory where the sparse index is stored. If not provided, the index is built from scratch. Useful for quickly benchmarking reranking models.')
    parser.add_argument('--no-chunking', action='store_true', help='Disable chunking of long documents')
    parser.add_argument('--no-flops', action='store_true', help='Disable computing flops / sparsity stats')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max supported length for embedding model')
    args = parser.parse_args()
    args.tokenizer = args.tokenizer or args.model

    # Resolve paths to absolute
    script_dir = os.path.dirname(os.path.realpath(__file__))
    args.cache_dir = os.path.abspath(os.path.join(script_dir, args.cache_dir))
    os.makedirs(args.cache_dir, exist_ok=True)
    args.jsonl_log = os.path.abspath(os.path.join(script_dir, args.jsonl_log))

    # avoid footgun - you might think you were re-computing embeddings but accidentially specify a reused index
    assert args.sparse_index_dir is None or args.rerank_model is not None, "You probably don't want to reuse a index unless you're testing reranking"

    # Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().addFilter(UnpickleableLoggingFilter())
    logger.setLevel(logging.INFO)
    logger.info(f"Running benchmark on dataset: {args.dataset} with split: {args.split}")
    logger.info(f"Data will be downloaded in: {args.cache_dir!r}")
    if args.note:
        logger.info(f"Note: {args.note}")
    logger.info(f"Results will be logged in: {args.jsonl_log!r}")

    #download & load data
    if args.dataset == 'fermi' or args.dataset.lower() == 'fermibench':
        args.dataset = 'atomic-canyon/FermiBench'
    if os.path.isdir(args.dataset):
        data_path = args.dataset
    elif args.dataset.startswith('http://') or args.dataset.startswith('https://'):
        data_path = util.download_and_unzip(args.dataset, args.cache_dir)
    elif args.dataset.count("/") == 1:
        data_path = os.path.join(args.cache_dir, args.dataset.split("/")[-1])
        if not os.path.exists(data_path):
            snapshot_download(repo_id=args.dataset, local_dir=data_path, repo_type='dataset')
    else:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
        data_path = util.download_and_unzip(url, args.cache_dir)

    # load the dataset
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=args.split)

    # build an adapter to emulate bier's interface
    model = SparseSearchAdapter(
        args.model,
        args.tokenizer,
        rerank_model=args.rerank_model,
        sparse_index_dir=args.sparse_index_dir,
        max_tokens=args.max_tokens,
        compute_flops=not args.no_flops,
        do_chunking=not args.no_chunking
    )

    # retrieve the results for each query in the dataset
    retriever = EvaluateRetrieval(model, k_values=[1,3,5,10,100,1000])
    results = retriever.retrieve(corpus, queries)

    # evaluate those results
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    hr = hr_k(qrels, results, retriever.k_values)

    # dump the results to the screen and logfile
    res_json = json.dumps(dict(
        ndcg=ndcg,
        map=_map,
        recall=recall,
        precision=precision,
        mrr=mrr,
        hr=hr,
        args=vars(args),
        eval_time=datetime.now().isoformat(),
        flops=model.flops if not args.no_flops else None,
        avg_doc_sparsity=model.avg_doc_sparsity,
        avg_query_sparsity=model.avg_query_sparsity,
    ))
    print(res_json)
    with open(args.jsonl_log, 'a') as file:
        file.write(res_json + '\n')

if __name__ == '__main__':
    main()

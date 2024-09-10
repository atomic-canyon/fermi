import json
import gzip
import os
import tempfile
import numpy as np
import plyvel
from decouple import config
from transformers import AutoTokenizer
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

# pull the number of processes to use in parallel from env var NUM_PROC (or default 16)
NUM_PROC = config('NUM_PROC', default=16, cast=int)


#modified from https://github.com/castorini/pyserini/blob/d4d936851c284838551ef2ebb69c0a298ffb9c78/pyserini/encode/_splade.py#L8
def output_to_weight_dicts(batch_aggregated_logits, reverse_voc):
    to_return = []
    for aggregated_logits in batch_aggregated_logits:
        col = np.nonzero(aggregated_logits)[0]
        weights = aggregated_logits[col]
        d = {reverse_voc[k]: float(v) for k, v in zip(list(col), list(weights))}
        to_return.append(d)
    return to_return

#modified from https://github.com/castorini/pyserini/blob/d4d936851c284838551ef2ebb69c0a298ffb9c78/pyserini/encode/_splade.py#L8
def get_encoded_query_token_weight_dicts(tok_weights, scale_factor=50.0, max_terms=1024):
    # two hacks here
    # *  pyserini by default cannot feed weights of terms to lucene
    # *  pyserini's default config only handles up to 1024 unique terms

    # To address this,
    #  Anserini first creates fake documents from JSON weight files (e.g.,
    #  {"hello": 3}) by repeating the term (e.g., "helo hello hello") and then
    #  indexes these documents as regular documents.

    # Additionally, we sort the terms by their weight and take only the top 1024
    # If you see a lot of warnings about truncation, your model is actually not
    # very sparse and you should investigate further.

    to_return = []
    for _tok_weight in tok_weights:
        _weights = {}
        for token, weight in _tok_weight.items():
            weight_quanted = round(weight * scale_factor)
            if weight_quanted == 0: continue #skip zero weights
            _weights[token] = weight_quanted
        if len(_weights.keys()) > max_terms:
            _weights = list(_weights.items())
            logger.warn(f"WARNING: number of terms {len(_weights)} exceeds 1024, truncating to 1024")
            _weights.sort(key=lambda x: x[1], reverse=True)
            _weights = _weights[:max_terms] #pyserini will fail if total is over 1024
        assert len(_weights) <= max_terms, f"total: {len(_weights)} exceeds 1024 it is going to fail in pyserini {_weights}"
        _weights = dict(_weights)
        to_return.append(_weights)
    return to_return

# See https://github.com/castorini/pyserini/blob/master/docs/experiments-unicoil.md for more info about how this sparse lucene works
def save_sparse_collection(ds, collection_folder, tokenizer, text_key):
    if type(tokenizer) == str:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
    def _format_doc(ex):
        # format the document in the specific json format supported by anserini
        vec = np.array(ex['embeddings'])
        rep = output_to_weight_dicts([vec], reverse_voc)
        rep = get_encoded_query_token_weight_dicts(rep)[0]
        item = dict(
            id=f"{ex['docKey']}_{ex['idx']}",
            contents=ex[text_key],
            vector=rep,
        )
        return dict(json_data=json.dumps(item))
    fnds = ds.map(_format_doc, remove_columns=ds.column_names, num_proc=NUM_PROC, desc='converting to expected sparse index format')

    #need to save file in jsonl supported by https://github.com/castorini/anserini/blob/f6f0dd6dc4cdbc77b9b6fac10ed21dd6bed76d13/src/main/java/io/anserini/collection/JsonVectorCollection.java#L28
    fpath = os.path.join(collection_folder, "split00.jsonl.gz")
    with gzip.open(fpath, 'wt', encoding='UTF-8', compresslevel=2) as file:
        for js in tqdm(fnds, desc="writing to split00.jsonl.gz"):
            file.write(js['json_data'] + '\n')

def save_sparse_collection_text(ds, collection_folder, text_key):
    # Save the raw text of each chunk to LevelDB for debugging or reranking
    db = plyvel.DB(collection_folder + "/collection.lvldb", create_if_missing=True)
    def make_chunk_lut(ex):
        db.put(f"{ex['docKey']}_{ex['idx']}".encode('utf-8'), ex[text_key].encode('utf-8'))
        return {}
    ds.map(make_chunk_lut, num_proc=1, load_from_cache_file=False, desc=f'save {text_key} to lvldb')

open_dbs = {} #XXX: we don't close the DBs. If used outside of testing fix.
def lookup_sparse_collection_text(lucene_index_path, lucene_docid):
    assert isinstance(lucene_docid, str), f"lucene_docid should be str but got {type(lucene_docid)}"
    db_path = os.path.abspath(os.path.join(lucene_index_path, '../collection/collection.lvldb')) # Kinda hacks but works
    if not os.path.exists(db_path):
        return None
    if db_path not in open_dbs:
        open_dbs[db_path] = plyvel.DB(db_path)
    db = open_dbs[db_path]
    return db.get(lucene_docid.encode('utf-8')).decode('utf-8')

def run_pyserini_index_sparse(collection_folder, index_path, threads=12):
    from jnius import autoclass
    JIndexCollection = autoclass('io.anserini.index.IndexCollection')
    args = [
        '-collection', 'JsonVectorCollection',
        '-input', collection_folder,
        '-index', index_path,
        '-generator', 'DefaultLuceneDocumentGenerator',
        '-threads', str(int(threads)),
        '-impact',
        '-pretokenized',
    ]
    JIndexCollection.main(args)

def create_sparse_index(ds, model_name, text_key='text', save_text=False, save_dir=None):
    temp_dir = save_dir or tempfile.mkdtemp()
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    logger.info(f'sparse index being created in dir {temp_dir!r}')
    collection_path = os.path.join(temp_dir, "collection")
    index_path = os.path.join(temp_dir, "index")
    if os.path.exists(index_path):
        # if you're testing re-rankers, you do not want to rebuild the sparse index every time
        # thus if the index dir already exists, just quit out early
        logger.info(f'index path "{index_path}" already exists, returning')
        return index_path
    os.mkdir(collection_path)
    os.mkdir(index_path)

    # add an index column from 0..n documents
    idx = -1
    def _add_index(example):
        nonlocal idx
        idx +=1
        return dict(idx=idx)
    ds = ds.map(_add_index, num_proc=1, desc='add idx column for sparse index format')

    save_sparse_collection(ds, collection_path, model_name, text_key)
    run_pyserini_index_sparse(collection_path, index_path)
    if save_text:
        save_sparse_collection_text(ds, collection_path, text_key)
    logger.info(f'sparse index create finished at {index_path!r}')
    return index_path

import json
import os
import numpy as np
from decouple import config
from transformers import AutoTokenizer
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, model
)

logger = logging.getLogger(__name__)

NUM_PROC = config('NUM_PROC', default=16, cast=int)
MILVUS_HOST = config('MILVUS_HOST', default='localhost')
MILVUS_PORT = config('MILVUS_PORT', default=19530, cast=int)

class MilvusSparseAdapter:
    def __init__(self, collection_name: str = "sparse_documents", 
                 host: str = MILVUS_HOST, port: int = MILVUS_PORT):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.collection = None
        self.splade_ef = None
        self._connect()
        
    def _connect(self):
        """Connect to Milvus server"""
        connections.connect("default", host=self.host, port=self.port)
        logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        
    def create_collection(self, drop_if_exists: bool = True):
        """Create a collection for sparse documents"""
        if drop_if_exists and utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"Dropped existing collection: {self.collection_name}")
            
        # Define schema for sparse vectors
        fields = [
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        
        schema = CollectionSchema(fields, description="Sparse document collection")
        self.collection = Collection(self.collection_name, schema)
        
        # Create index for sparse vectors - no drop ratio for perfect recall
        index_params = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",  # Inner Product for sparse vectors
        }
        self.collection.create_index("sparse_vector", index_params)
        logger.info(f"Created collection: {self.collection_name}")
        
    def load_collection(self):
        """Load collection into memory"""
        if not self.collection:
            self.collection = Collection(self.collection_name)
        self.collection.load()
        logger.info(f"Loaded collection: {self.collection_name}")
        
    def init_splade_model(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        """Initialize SPLADE embedding function"""
        self.splade_ef = model.sparse.SpladeEmbeddingFunction(
            model_name=model_name,
            device="cuda" if config('CUDA_AVAILABLE', default=True, cast=bool) else "cpu"
        )
        logger.info(f"Initialized SPLADE model: {model_name}")

def numpy_sparse_to_milvus_sparse(sparse_vector: np.ndarray, reverse_vocab: Dict[int, str]) -> Dict:
    """Convert numpy sparse vector to Milvus sparse format"""
    indices = np.nonzero(sparse_vector)[0]
    values = sparse_vector[indices]
    
    # Milvus expects {index: value} format
    return {int(idx): float(val) for idx, val in zip(indices, values)}

def create_milvus_sparse_index(ds, model_name: str, text_key: str = 'text', 
                              collection_name: str = "sparse_documents", 
                              save_dir: Optional[str] = None) -> str:
    """Create Milvus sparse collection from dataset"""
    
    # Initialize Milvus adapter
    adapter = MilvusSparseAdapter(collection_name)
    adapter.create_collection()
    adapter.init_splade_model(model_name)
    
    # Prepare tokenizer for sparse vector conversion
    if isinstance(model_name, str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = model_name
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
    
    # Process dataset in batches
    batch_size = 100
    total_docs = len(ds)
    
    for i in tqdm(range(0, total_docs, batch_size), desc="Inserting documents"):
        batch = ds.select(range(i, min(i + batch_size, total_docs)))
        
        doc_ids = []
        texts = []
        sparse_vectors = []
        
        for example in batch:
            doc_id = f"{example['docKey']}_{example.get('idx', 0)}"
            text = example[text_key]
            
            # Convert numpy sparse to Milvus format
            sparse_vec = numpy_sparse_to_milvus_sparse(
                np.array(example['embeddings']), 
                reverse_voc
            )
            
            doc_ids.append(doc_id)
            texts.append(text)
            sparse_vectors.append(sparse_vec)
        
        # Insert batch into Milvus
        data = [doc_ids, texts, sparse_vectors]
        adapter.collection.insert(data)
    
    adapter.collection.flush()
    adapter.load_collection()
    
    logger.info(f"Created Milvus sparse collection with {total_docs} documents")
    return collection_name

class MilvusImpactSearcher:
    """Milvus-based replacement for LuceneImpactSearcher"""
    
    def __init__(self, collection_name: str, splade_model: str = "naver/splade-cocondenser-ensembledistil"):
        self.collection_name = collection_name
        self.collection = Collection(collection_name)
        self.collection.load()
        
        # Initialize SPLADE for query encoding
        self.splade_ef = model.sparse.SpladeEmbeddingFunction(
            model_name=splade_model,
            device="cuda" if config('CUDA_AVAILABLE', default=True, cast=bool) else "cpu"
        )
        
        # Initialize tokenizer for compatibility
        self.tokenizer = AutoTokenizer.from_pretrained(splade_model)
        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}
        
    def search(self, query: str, k: int = 10) -> List:
        """Search for documents using sparse vectors"""
        # Encode query using SPLADE
        query_embeddings = self.splade_ef.encode_queries([query])
        
        # Convert CSR sparse array to Milvus format
        # For csr_array, access indices and data directly for the first row
        query_sparse = {int(idx): float(val) for idx, val in zip(query_embeddings.indices, query_embeddings.data)}
        
        # Search in Milvus - no drop ratio for perfect recall
        search_params = {
            "metric_type": "IP",
        }
        
        results = self.collection.search(
            data=[query_sparse],
            anns_field="sparse_vector",
            param=search_params,
            limit=k,
            output_fields=["doc_id", "text"]
        )
        
        # Convert results to compatible format
        hits = []
        for hit in results[0]:  # First query results
            hit_obj = type('Hit', (), {})()
            hit_obj.docid = hit.entity.get("doc_id")
            hit_obj.score = hit.score
            hits.append(hit_obj)
            
        return hits

def lookup_milvus_collection_text(collection_name: str, doc_id: str) -> Optional[str]:
    """Lookup document text by ID from Milvus collection"""
    collection = Collection(collection_name)
    
    # Query for specific document
    results = collection.query(
        expr=f'doc_id == "{doc_id}"',
        output_fields=["text"]
    )
    
    if results:
        return results[0]["text"]
    return None
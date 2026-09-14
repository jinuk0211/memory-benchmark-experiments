"""Fetch the pinned original embedding model; existing inference service is reused."""
from pathlib import Path
import importlib
import json
from huggingface_hub import snapshot_download

for name in ('numpy','nltk','rank_bm25','transformers','sentence_transformers'):
    mod=importlib.import_module(name)
    print(name,getattr(mod,'__version__','available'),flush=True)
path=snapshot_download('Qwen/Qwen3-Embedding-0.6B',revision='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3',
                       cache_dir='/workspace/.hf_home/hub',
                       ignore_patterns=['*.md','.gitattributes','*.png','*.jpg','onnx/*','openvino/*'])
print(json.dumps({'embedding_path':path,'ready':True}),flush=True)

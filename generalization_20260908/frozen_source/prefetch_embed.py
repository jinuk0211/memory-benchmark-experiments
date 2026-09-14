import os, time
os.environ['HF_HOME'] = '/workspace/.hf_home'
os.environ['HF_HUB_DISABLE_XET'] = '0'
from huggingface_hub import snapshot_download
started = time.time()
print('XET_EMBED_START',flush=True)
print(snapshot_download('Qwen/Qwen3-Embedding-0.6B',ignore_patterns=['*.bin','onnx/*']),flush=True)
print('XET_EMBED_READY',time.time()-started,flush=True)

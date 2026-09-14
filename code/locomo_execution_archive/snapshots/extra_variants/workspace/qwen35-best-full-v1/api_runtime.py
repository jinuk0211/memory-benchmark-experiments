"""Metered API execution of frozen recursive constructors on Qwen3.5-9B."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import threading
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
import refine as core
from evidence_utility import POLICY, answer_token_request


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def validate_completion(body, prompts, nll=False):
    choices=sorted(body['choices'],key=lambda c:c['index'])
    assert [c['index'] for c in choices]==list(range(len(prompts)))
    usage=body.get('usage')
    if not isinstance(usage,dict):raise ValueError('Model did not report actual usage')
    for field in ('prompt_tokens','completion_tokens','total_tokens'):
        if not isinstance(usage.get(field),int) or usage[field]<0:raise ValueError('Invalid usage')
    assert usage['prompt_tokens']==sum(len(p) for p in prompts)
    assert usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
    output_tokens=0
    for choice,prompt in zip(choices,prompts):
        if choice.get('prompt_token_ids')!=prompt:raise ValueError('Server changed prompt token IDs')
        if not isinstance(choice.get('token_ids'),list):raise ValueError('Missing output token IDs')
        if choice.get('finish_reason') not in ('stop','length'):raise ValueError('Nonterminal model response')
        if not isinstance(choice.get('text'),str):raise ValueError('Missing generated text')
        output_tokens+=len(choice['token_ids'])
        if nll and len(choice.get('prompt_logprobs') or [])!=len(prompt):
            raise ValueError('Missing prompt likelihood positions')
    assert output_tokens==usage['completion_tokens']
    return choices


def target_likelihood(choice,start,targets):
    assert choice['prompt_token_ids'][start:]==list(targets)
    values=[]
    for i,token in enumerate(targets):
        row=choice['prompt_logprobs'][start+i]
        value=row.get(str(token),row.get(token)) if row else None
        if value is None:raise ValueError('Actual target token probability missing')
        lp=float(value['logprob'])
        if not math.isfinite(lp) or lp>1e-5:raise ValueError('Invalid log probability')
        values.append(lp)
    return {'mean_logprob':sum(values)/len(values),'sum_logprob':sum(values),
            'answer_tokens':len(values),'token_logprobs':values,'target_token_ids':list(targets),
            'answer_start':start,'request_token_sha256':core.digest(choice['prompt_token_ids'])}


class Runtime:
    def __init__(self,args,protocol):
        from transformers import AutoTokenizer
        self.args=args; self.protocol=protocol;self.np=np
        self.cache=Path(args.out)/'cache';self.cache.mkdir(parents=True,exist_ok=True)
        self.model_meta=protocol['generation_model'];self.embed_meta=protocol['embedding_model']
        self.tok=AutoTokenizer.from_pretrained(self.model_meta['path'],local_files_only=True)
        self.ntok=lru_cache(maxsize=100000)(lambda s:len(self.tok.encode(s,add_special_tokens=False)))
        self.phase='initialization';self.conversation=None
        self.lock=threading.Lock();self.embed=None
        self.usage_path=Path(args.out)/'usage.jsonl'

    def log(self,row):
        with self.lock:
            with self.usage_path.open('a',encoding='utf8') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()

    def request(self,prompts,max_tokens,seed,nll=False,keys=None):
        logical=core.digest([self.model_meta,prompts,max_tokens,seed,nll])
        payload={'model':self.args.model,'prompt':prompts,'temperature':0,'top_p':1,
                 'max_tokens':max_tokens,'seed':seed,'return_token_ids':True,'add_special_tokens':False}
        if nll:payload['prompt_logprobs']=0
        phase,conversation=self.phase,self.conversation
        encoded=json.dumps(payload).encode()
        for attempt in range(3):
            started=time.time();request_id=str(uuid.uuid4())
            event={'request_id':request_id,'logical_request_id':logical,'attempt':attempt,
                   'phase':phase,'conversation':conversation,'kind':'source_likelihood' if nll else 'generation',
                   'model':self.args.model,'seed':seed,'temperature':0,'max_tokens':max_tokens,
                   'prompt_count':len(prompts),'prompt_lengths':[len(p) for p in prompts],
                   'prompt_token_sha256':[core.digest(p) for p in prompts],'cache_keys':keys,
                   'started_at':started,'request_sha256':hashlib.sha256(encoded).hexdigest()}
            response=None
            try:
                req=urllib.request.Request(self.args.base_url+'/v1/completions',data=encoded,
                                          headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=1800) as f:
                    response=json.load(f)
                choices=validate_completion(response,prompts,nll)
                self.log({**event,'success':True,'response_id':response.get('id'),'usage':response['usage'],
                          'finish_reasons':[c['finish_reason'] for c in choices],
                          'output_lengths':[len(c['token_ids']) for c in choices],
                          'seconds':time.time()-started,'completed_at':time.time()})
                return choices,request_id
            except Exception as exc:
                detail=str(exc)[:2000]
                if isinstance(exc,urllib.error.HTTPError):detail=exc.read().decode(errors='replace')[:2000]
                self.log({**event,'success':False,'error_type':type(exc).__name__,'error':detail,
                          'usage':response.get('usage') if response else None,
                          'seconds':time.time()-started,'completed_at':time.time(),
                          'unknown_usage':response is None})
                if response is not None:core.save(Path(self.args.out)/'failures'/f'{request_id}.json',response)
                retry=isinstance(exc,(urllib.error.URLError,TimeoutError))
                if isinstance(exc,urllib.error.HTTPError):retry=exc.code in (408,429,500,502,503,504)
                if not retry or attempt==2:raise
                time.sleep(min(5*(attempt+1),10))

    def generate(self,system,users,max_tokens=512,seed=None):
        seed=self.args.seed if seed is None else seed
        keys=[];pending={};results={}
        for user in users:
            key=core.digest(['qwen35_api_v1',self.model_meta,seed,system,user,max_tokens,False])
            keys.append(key);path=self.cache/'generations'/f'{key}.json'
            if path.exists():results[key]=read(path)['text'];continue
            if key in pending:continue
            prompt=self.tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':user}],
                tokenize=True,add_generation_prompt=True,enable_thinking=False)
            if len(prompt)+max_tokens>8192:raise ValueError('Frozen constructor context cap exceeded')
            pending[key]=(list(prompt),path)
        entries=list(pending.items())
        batches=[entries[i:i+self.args.api_batch] for i in range(0,len(entries),self.args.api_batch)]
        def run(batch):
            prompts=[v[0] for k,v in batch]
            choices,rid=self.request(prompts,max_tokens,seed,keys=[k for k,v in batch])
            out=[]
            for (key,(prompt,path)),choice in zip(batch,choices):
                value={'text':choice['text'].strip(),'finish_reason':choice['finish_reason'],
                       'input_tokens':len(prompt),'output_tokens':len(choice['token_ids']),
                       'token_ids':choice['token_ids'],'request_id':rid,
                       'prompt_token_sha256':core.digest(prompt)}
                core.save(path,value);out.append((key,value['text']))
            return out
        if batches:
            with ThreadPoolExecutor(max_workers=self.args.api_workers) as pool:
                for i,batch_result in enumerate(pool.map(run,batches)):
                    results.update(batch_result)
                    print(f'GEN phase={self.phase} conv={self.conversation} batch={i+1}/{len(batches)} cap={max_tokens}',flush=True)
        return [results[key] for key in keys]

    def encode(self,texts,query=False):
        key=core.digest(['qwen_embedding_cpu_fp32_v1',self.embed_meta,texts,query,self.args.embed_batch_size])
        path=self.cache/'embeddings'/f'{key}.npy'
        if path.exists():return np.load(path)
        if self.embed is None:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(self.args.cpu_threads)
            self.embed=SentenceTransformer(self.embed_meta['path'],device='cpu',local_files_only=True)
        prepared=['Instruct: Retrieve relevant conversation memories to answer the question.\nQuery: '+t for t in texts] if query else texts
        started=time.time()
        actual=[];lengths=[];max_len=self.embed.max_seq_length
        for t in prepared:
            ids=self.embed.tokenizer.encode(t,add_special_tokens=True,truncation=True,max_length=max_len)
            actual.append(len(ids));lengths.append(len(self.embed.tokenizer.encode(t,add_special_tokens=True,truncation=False)))
        values=self.embed.encode(prepared,batch_size=self.args.embed_batch_size,normalize_embeddings=True,show_progress_bar=False)
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.tmp')
        with temp.open('wb') as f:np.save(f,values)
        temp.replace(path)
        info={'request_id':str(uuid.uuid4()),'phase':self.phase,'conversation':self.conversation,
              'kind':'embedding','model':self.embed_meta['model'],'revision':self.embed_meta['revision'],
              'query':query,'cache_key':key,'records':len(texts),'input_tokens':sum(actual),
              'untruncated_input_tokens':sum(lengths),'truncated_inputs':sum(a<b for a,b in zip(actual,lengths)),
              'token_lengths':actual,'device':'cpu','dtype':str(values.dtype),'seconds':time.time()-started,
              'success':True,'completed_at':time.time()}
        self.log(info);core.save(path.with_suffix('.json'),info)
        return values


class Scorer:
    def __init__(self,rt):
        self.rt=rt;self.tok=rt.tok;self.ntok=rt.ntok

    def score(self,jobs):
        pending={}
        for job in jobs:
            request,start,target=answer_token_request(self.tok,job['user'],job['answer'])
            ids=request['prompt_token_ids']
            if len(ids)+1>8192:raise ValueError('Frozen NLL context cap exceeded')
            key=core.digest(['nll_qwen35_v1',self.rt.model_meta,POLICY['seed'],ids,start,target])
            path=self.rt.cache/'nll'/f'{key}.json'
            if path.exists():job['score']=read(path)
            else:
                item=pending.setdefault(key,{'path':path,'ids':ids,'start':start,'target':target,'jobs':[]})
                item['jobs'].append(job)
        entries=list(pending.items());batches=[entries[i:i+self.rt.args.api_batch] for i in range(0,len(entries),self.rt.args.api_batch)]
        def run(batch):
            choices,rid=self.rt.request([v['ids'] for k,v in batch],1,POLICY['seed'],True,[k for k,v in batch])
            for (key,item),choice in zip(batch,choices):
                value={**target_likelihood(choice,item['start'],item['target']),
                       'request_id':rid,'input_tokens':len(item['ids']),'output_tokens':len(choice['token_ids'])}
                core.save(item['path'],value)
                for job in item['jobs']:job['score']=value
        with ThreadPoolExecutor(max_workers=self.rt.args.api_workers) as pool:
            list(pool.map(run,batches))

    def generate(self,controls):
        names=list(controls)
        texts=self.rt.generate(core.READER,[controls[n]['user'] for n in names],POLICY['max_answer_tokens'],seed=POLICY['seed'])
        return {name:{'text':text,'source_ids':controls[name]['source_ids'],
                      'context_tokens':controls[name]['context_tokens'],
                      'source_answer_f1':core.generic_f1(text,controls[name]['answer'])}
                for name,text in zip(names,texts)}

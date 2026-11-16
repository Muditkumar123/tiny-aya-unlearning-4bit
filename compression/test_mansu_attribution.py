"""CPU CircuitKIT edge-attribution integration check with synthetic tokens; no downloads."""
import torch
torch.set_num_threads(1)
torch.manual_seed(0)
import circuitkit  # Install its model compatibility patches before TransformerLens.
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from transformer_lens import HookedTransformer, HookedTransformerConfig
from circuitkit.backends.eap.attribute import attribute
from circuitkit.backends.eap.graph import Graph, MLPNode
from circuitkit.data.normalized import ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset
from circuitkit.data.normalized_task import NormalizedTaskSpec
import tempfile
raw=Tokenizer(WordLevel({'[UNK]':0,'[PAD]':1,'[BOS]':2,'[EOS]':3,'foo':4,'bar':5,'A':6,'B':7},unk_token='[UNK]'));raw.pre_tokenizer=WhitespaceSplit()
tok=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',pad_token='[PAD]',bos_token='[BOS]',eos_token='[EOS]')
token_dir=tempfile.TemporaryDirectory()
tok.save_pretrained(token_dir.name)
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(token_dir.name,local_files_only=True)
cfg=HookedTransformerConfig(n_layers=2,n_heads=2,d_model=16,d_head=8,d_mlp=32,n_ctx=32,d_vocab=8,act_fn='relu',device='cpu',use_attn_result=True,use_split_qkv_input=True,use_hook_mlp_in=True)
model=HookedTransformer(cfg,tokenizer=tok)
record=ContrastiveRecord(record_id='toy',clean_prompt='foo',clean_answer='A',corrupt_prompt='bar',corrupt_answer='B',contrast_source=ContrastSource.EXTERNAL)
ds=NormalizedDataset(name='toy',shape=DatasetShape.FORGET_RETAIN,records=[record],source='synthetic')
with tempfile.TemporaryDirectory() as d:
 task=NormalizedTaskSpec(ds,name='toy',cache_dir=d)
 loader=task.build_dataloader(model,{'algorithm':'eap-ig','level':'node','n_examples':1,'batch_size':1},'cpu')
 graph=Graph.from_model(model,node_scores=True,neuron_level=False,mlp_hook='mlp_out')
 attribute(model,graph,loader,task.metric_fn(),method='EAP-IG-inputs',ig_steps=2,quiet=True)
 edges={n:float(e.score) for n,e in graph.edges.items() if isinstance(e.child,MLPNode)}
 assert edges and sum(abs(v) for v in edges.values())>0
 print('Tiny CircuitKIT EAP-IG integration passed; no download or GPU.')

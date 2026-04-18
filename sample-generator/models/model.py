import torch as th

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = ''

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    tokenizer.paddin_side = 'left'
    return tokenizer

def get_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype = th.bfloat16,
        attn_implementation = "flash_attention_2")
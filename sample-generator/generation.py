import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch as th
import numpy as np
import pandas as pd
import json 

from models.model import get_tokenizer, get_model

FEATURE_NAMES = [
    "user_query",
    "temp_media_estate",
    "temp_media_inverno",
    "temp_media_primavera",
    "temp_media_autunno",
    "giorni_sole_anno",
    "mm_pioggia_anno",
    "umidita_media",
    "clima_type",
    "qualita_aria_aqi",
    "mare",
    "lago",
    "fiume",
    "montagna",
    "collina",
    "n_spiagge_libere",
    "km_parchi_nazionali",
    "altitudine_media",
    "pct_verde_urbano",
    "km_costa",
    "tipo_cucina",
    "n_ristoranti_per_Xab",
    "mercati_locali",
    "vita_notturna_score",
    "n_bar_per_1000ab",
    "n_musei",
    "n_siti_unesco",
    "patrimonio_storico_score",
    "n_teatri_cinema",
    "n_gallerie_arte",
    "lingua_ufficiale",
    "km_piste_ciclabili",
    "km_sentieri_trekking",
    "surf_score",
    "n_abitanti",
    "affluenza_taxi",
    "intesita_transp_pubb",
]

user_queries = []
max_user_queries = 1500
df = pd.DataFrame(columns=FEATURE_NAMES)

def generate_txt(model, tokenizer, query, max_tokens = 128):
    device = next(model.parameters()).device()

    if query:
        prompt = 'Generate a user query'
    else : 
        prompt = f"{query}"

    inputs = tokenizer(prompt, return_tensors = 'pt').to(device)

    with th.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens = max_tokens,
            do_sample = False,
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = tokenizer.eos_token_id
        )
    
    new_token = out[0, inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_token, skip_special_tokens = True)

def safe_parse_llm_json(row_output):
    raw_output = row_output.strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Nessun oggetto JSON trovato nell'output del modello.")
        return json.loads(raw_output[start:end + 1])

def llm_json(model, tokenizer, user_query, df):
    raw_output = generate_txt(
        model = model,
        tokenizer=tokenizer,
        query=None,
    )

    data = safe_parse_llm_json(raw_output)
    data["user_query"] = user_query

    df.loc[len(df)] = {col : data.get(col, None) for col in FEATURE_NAMES}
    return df


model = get_model()
tokenizer = get_tokenizer()

for _ in range(max_user_queries):
    user_queries.append(generate_txt(
        model = model,
        tokenizer=tokenizer,
        query=None,
    ))

for user_q in user_queries: 
    df = llm_json(model, tokenizer, user_q, df)

df.to_csv("query_dataset_llm.csv", index=False)
import os
import json
import pickle
import gzip
import pandas as pd
import os
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import argparse

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def load_meta(data_path:str, datamap:dict):
    meta = {}
    
    meta_file = os.path.join(data_path, "meta.json.gz")

    for d in parse(meta_file):
        if d["asin"] in datamap['item2id'].keys():
            meta[d["asin"]] = d
            item_desc = ""
            if 'title' in d.keys():
                item_desc += 'Title: '
                item_desc += d['title']
            if 'description' in d.keys():
                item_desc += 'Description: '
                item_desc += d['description']
            if 'feature' in d.keys():
                item_desc += 'Features: '
                item_desc += ", ".join(d['feature'])
            if 'categories' in d.keys():
                item_desc += 'Categories: '
                item_desc += ", ".join(d['categories'][0])
                
            meta[d["asin"]].update({'item_desc': item_desc})
    return meta


def main(model_name, data_name, batch, device):

    # load the data
    if "beauty" in data_name:
        data_path = os.path.join("data", "beauty")
    elif "toy" in data_name:
        data_path = os.path.join("data", "toys")
    else:
        raise ValueError("data_name should be either 'beauty' or 'toy'")

    print("Loading data from {}".format(data_path))
    datamap = json.load(open(os.path.join(data_path, "datamaps.json"), "r"))
    item2id = datamap['item2id']
    id2item = datamap['id2item']
    user2id = datamap['user2id']
    id2user = datamap['id2user']
    sorted_item_ids = list(id2item.keys())
    sorted_user_ids = list(id2user.keys())
    meta = load_meta(data_path, datamap)

    # load the model
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # get the item embeddings
    print("Getting item embeddings")
    item2emb = {}

    for item_id in tqdm(range(0, len(sorted_item_ids), batch)):
        batch_item = sorted_item_ids[item_id:item_id+batch]
        batch_desc = [meta[id2item[i]]['item_desc'] for i in batch_item]
        batch_input = tokenizer(batch_desc, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

        batch_output = model(**batch_input)
        batch_cls = batch_output[0][:, 0, :].detach().cpu()

        for batch_id, item_id in enumerate(batch_item):
            item2emb[item_id] = torch.Tensor(batch_cls[batch_id])
            
            assert item2emb[item_id].shape == (512,)

    # get the user embeddings
    print("Getting user embeddings")
    user2emb = {}

    for line in open(os.path.join(data_path, "sequential_data.txt"), "r"):
        l = line.strip().split(" ")
        user_id = l[0]
        item_id = l[1:-2]
        item_emb = [item2emb[i] for i in item_id]
        user_emb = torch.mean(torch.stack(item_emb), dim=0)
        user2emb[user_id] = user_emb

        assert user2emb[user_id].shape == (512,)

    # concatenate the user and item embeddings
    user_emb = torch.stack([user2emb[id] for id in sorted_user_ids])
    item_emb = torch.stack([item2emb[id] for id in sorted_item_ids])
    token_emb = torch.cat([user_emb, item_emb], dim=0)

    # print shape of embedding
    print(user_emb.shape)
    print(item_emb.shape)
    print(token_emb.shape)


    embedding_dict = {}

    embedding_dict['token_embedding'] = token_emb
    embedding_dict['user_token_embedding'] = {f"user_{id}": user2emb[id] for id in sorted_user_ids}
    embedding_dict['item_token_embedding'] = {f"item_{id}": item2emb[id] for id in sorted_item_ids}


    # save the embedding
    pickle.dump(embedding_dict, open(os.path.join("initialize",f"{data_name}_bert.pkl"), "wb"))
    
    return 1


if __name__ == '__main__':
    model_name = ("google/bert_uncased_L-8_H-512_A-8")
    data_name = "toy"
    batch = 32
    device = "cuda:1"

    main(model_name=model_name, data_name=data_name, batch=batch, device=device)
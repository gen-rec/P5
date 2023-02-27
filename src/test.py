
from transformers import T5TokenizerFast
from pathlib import Path
from param import parse_args
import os.path
from pretrain_model import P5Pretraining
from transformers import T5Config
import torch
from src.all_amazon_templates import all_tasks
import gzip

import pickle

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
import json

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def main(args):
    evaluator = P5Evaluator(args)
    evaluator.evalate()

    return 1



class P5Evaluator():

    def __init__(self,args) -> None:

        ## Load pretrained model and tokenizer
        project_dir = Path(__file__).resolve().parent.parent
        self.model = self.create_model(
            model_path=os.path.join(project_dir, args.load, 'BEST_EVAL_LOSS.pth'),
            args=args)
        self.tokenizer = self.create_tokenizer(
            tokenizer_path=os.path.join(project_dir, args.load, 'tokenizer-0/'))
        
        ## Load test data
        data_splits = load_pickle(os.path.join(project_dir, f'data/{args.test}/data_splits.pkl'))
        test_review_data = data_splits['test']
        print(f"Loaded {len(test_review_data)} test reviews")

        data_maps = load_json(os.path.join(project_dir, f'data/{args.test}/data_maps.json'))
        print(f"number of users {len(data_maps['user2id'])}")
        print(f"number of items {len(data_maps['item2id'])}")
    
    def evalate(self):
        pass


    def create_tokenizer(self, tokenizer_path:str):
        print(f"Loading Tokenizer from {tokenizer_path}..")
        return T5TokenizerFast.from_pretrained(tokenizer_path)

    def create_config(self, args):

        if 't5' in args.backbone:
            config_class = T5Config
        else:
            return None

        config = config_class.from_pretrained(args.backbone)
        config.dropout_rate = args.dropout
        config.dropout = args.dropout
        config.attention_dropout = args.dropout
        config.activation_dropout = args.dropout
        config.losses = args.losses

        return config
    
    def create_model(self, model_path, args):

        print(f"Loading Model from {args.load}..")
        
        config = self.create_config(args)
        model_class = P5Pretraining
        model_name = args.backbone
        model = model_class.from_pretrained(
            model_name,
            config=config
        )
        model.load_state_dict(torch.load(model_path), strict=False)
        if int(args.local_rank) != -1:
            model.to(f"cuda:{args.local_rank}")
        else:
            model.to("cpu")

        return model
    


    def test_rating():
        print("\ntesting Task 1: Rating Prediction")
        pass


    def test_sequential():
        print("\ntesting Task 2: Sequential Recommendation..")
        pass

    def test_explanation():
        print("\ntesting Task 3: Explanation Generationg")
        pass

    def test_review_related():
        print("\ntesting Task 4: Review Related")

    def test_direct():
        print("\ntesting Task 5: Direct Recommendation..")
        pass

    

if __name__ == "__main__":
    args = parse_args()
    main(args)
# import file from parent directory
import sys
import os.path
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import T5TokenizerFast
from pathlib import Path
from param import parse_args
from pretrain_model import P5Pretraining
from transformers import T5Config
import torch
from src.all_amazon_templates import all_tasks
import gzip
from src.pretrain_data import get_loader
from tqdm import tqdm
import pickle
import json
from src.utils import load_state_dict

from notebooks.evaluate.utils import rouge_score, bleu_score, unique_sentence_percent, root_mean_square_error, mean_absolute_error


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
    evaluator.evaluate()

    return 1



class P5Evaluator():

    def __init__(self, args) -> None:

        ## configure
        self.args = args
        project_dir = Path(__file__).resolve().parent.parent
        _, self.model_type, model_info, self.task_type = args.load.split('/') # snap/naive/beauty-small-42/task-1
        self.data_type, _, self.seed = model_info.split('-')
        self.output_dir = os.path.join(project_dir, 'output', self.model_type, model_info, self.task_type)
        os.makedirs(self.output_dir, exist_ok=True)

        ## Load pretrained model and tokenizer
        self.tokenizer = self.create_tokenizer(
            tokenizer_path= None if self.model_type == 'naive' else os.path.join(project_dir, args.load, 'tokenizer-0/'),
            args=args)
        self.model = self.create_model(
            model_path=os.path.join(project_dir, args.load, 'BEST_EVAL_LOSS.pth'),
            args=args)
        self.model.tokenizer = self.tokenizer

        ## Load test data
        data_splits = load_pickle(os.path.join(project_dir, f'data/{self.data_type}/rating_splits_augmented.pkl'))
        test_review_data = data_splits['test']
        print(f"\n\nLoaded {len(test_review_data)} test reviews")

        data_maps = load_json(os.path.join(project_dir, f'data/{self.data_type}/datamaps.json'))
        print(f"number of users {len(data_maps['user2id'])}")
        print(f"number of items {len(data_maps['item2id'])}")

        
        if self.data_type == 'yelp':
            self.test_task_list = {'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
                            'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10','2-11', '2-12'],
                            'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9'],
                            'review': ['4-1', '4-2'],
                            'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
                            }
        else:
            self.test_task_list = {'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10'],
                            'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12', '2-13'],
                            'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11', '3-12'],
                            'review': ['4-1', '4-2', '4-3', '4-4'],
                            'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7', '5-8']}

        self.sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

    
    def evaluate(self):
        if self.task_type == 'task-1':
            self.evaluate_task1()
        elif self.task_type == 'task-2':
            self.evaluate_task2()
        elif self.task_type == 'task-3':
            self.evaluate_task3()
        elif self.task_type == 'task-4':
            self.evaluate_task4()
        elif self.task_type == 'task-5':
            self.evaluate_task5()
        elif self.task_type == 'all':
            self.evaluate_task1()
            self.evaluate_task2()
            self.evaluate_task3()
            self.evaluate_task4()
            self.evaluate_task5()
        
        return 1
    

    def evaluate_task1(self):

        print("\ntesting Task 1: Rating Prediction")

        for task in self.test_task_list['rating']:
            test_loader = get_loader(
                args=self.args,
                task_list= {'rating': [task]},
                sample_numbers=self.sample_numbers,
                split=self.data_type,
                mode='test',
                batch_size=args.batch_size,
                workers=0,
                distributed=False,
                tokenizer=self.tokenizer,
            )

            ## Evaluate
            source_text, gt, pred = [], [], []
            for i, batch in enumerate(tqdm(test_loader)):
                with torch.no_grad():
                    results = self.model.generate_step(batch)

                    source_text.extend(batch['source_text'])
                    gt.extend(batch['target_text'])
                    pred.extend(results)
            assert len(source_text) == len(gt) == len(pred) == len(test_loader.dataset)

            ## Save results
            total = []
            for s, g, p in zip(source_text, gt, pred):
                total.append( {
                    'source_text': s,
                    'gt': g,
                    'pred': p
                })
            os.makedirs(os.path.join(self.output_dir, 'task-1', task), exist_ok=True)
            json.dump(total, open(os.path.join(self.output_dir, 'task-1', task, 'results.json'), 'w'), indent=4)

            ## Calculate metrics
            predicted_rating = [(float(r), float(p)) for (r, p) in zip(gt, pred) if p in [str(i/10.0) for i in list(range(10, 50))]]
            if predicted_rating:
                RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
                print('RMSE {:7.4f}'.format(RMSE))
                MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
                print('MAE {:7.4f}'.format(MAE))
            else:
                RMSE, MAE = -1, -1
            ## Save metrics
            with open(os.path.join(self.output_dir, 'task-1','metrics.tsv'), 'a') as f:
                f.write(f"{task}\t{RMSE}\t{MAE}")
        
        pass


    def evaluate_task2(self):
        print("\nTesting Task 2: Sequential Recommendation..")

        ## Load test data
        for task in self.test_task_list['sequential']:
            test_loader = get_loader(
                args=args,
                task_list= {'sequential': [task]},
                sample_numbers=self.sample_numbers,
                split=self.data_type,
                mode='test',
                batch_size=args.batch_size,
                workers=0,
                distributed=False,
                tokenizer=self.tokenizer,
            )

            ## Evaluate
            source_text, gt, pred = [], [], []
            for i, batch in enumerate(tqdm(test_loader)):
                with torch.no_grad():
                    results = self.model.generate_step(batch)

                    source_text.extend(batch['source_text'])
                    gt.extend(batch['target_text'])
                    pred.extend(results)
        
        pass

    def evaluate_task3():
        print("\nTesting Task 3: Explanation Generationg")
        pass

    def evaluate_task4():
        print("\nTesting Task 4: Review Related")

    def evaluate_task5():
        print("\nTesting Task 5: Direct Recommendation..")
        pass


    def create_tokenizer(self, tokenizer_path:str, args):
        if tokenizer_path is None:
            print(f"Loading Tokenizer from {args.backbone}..")
            return T5TokenizerFast.from_pretrained(args.backbone)
        else:
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

        print(f"Loading Model from {model_path}..")
        
        config = self.create_config(args)
        model_class = P5Pretraining
        model_name = args.backbone
        model = model_class.from_pretrained(
            model_name,
            config=config
        )
        model.resize_token_embeddings(self.tokenizer.vocab_size)
        state_dict = load_state_dict(model_path, 'cpu')
        model.load_state_dict(state_dict)
        if int(args.local_rank) != -1:
            model.to(f"cuda:{args.local_rank}")
        else:
            model.to("cpu")

        return model
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
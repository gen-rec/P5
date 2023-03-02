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
from notebooks.evaluate.metrics4rec import evaluate_all
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
        self.device = f"cuda:{args.local_rank}" if int(args.local_rank) != -1 else "cpu"
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

            # self.test_task_list = {'rating': ['1-1'],
            #                 'sequential': ['2-1'],
            #                 'explanation': ['3-1'],
            #                 'review': ['4-2',],
            #                 'traditional': ['5-1']}
            
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
        task_type = 'task-1'
        print("\nTesting Task 1: Rating Prediction")
        os.makedirs(os.path.join(self.output_dir, task_type), exist_ok=True)
        with open(os.path.join(self.output_dir,task_type,'metrics.tsv'), 'a') as f:
            f.write(f"task\tRMSE\tMAE\n")
        for task_num, task in enumerate(self.test_task_list['rating']):
            print(f"{task}: {task_num+1:>2d}/{len(self.test_task_list['rating'])}")
            test_loader = get_loader(
                args=self.args,
                task_list= {'rating': [task]},
                sample_numbers=self.sample_numbers,
                split=self.data_type,
                mode='test',
                batch_size=self.args.batch_size,
                workers=0,
                distributed=False,
                tokenizer=self.tokenizer,
            )

            ## Evaluate
            source_text, gt, pred = [], [], []
            for i, batch in enumerate(test_loader):
                with torch.no_grad():
                    results = self.model.generate(
                        batch['input_ids'].to(self.device),
                    )
                    generated = self.tokenizer.batch_decode(results, skip_special_tokens=True)
                    source_text.extend(batch['source_text'])
                    gt.extend(batch['target_text'])
                    pred.extend(generated)
            assert len(source_text) == len(gt) == len(pred) == len(test_loader.dataset)

            ## Save results
            self.save_results(source_text=source_text, gt=gt, pred=pred, task=task, task_type=task_type)

            ## Calculate metrics
            predicted_rating = [(float(r), float(p)) for (r, p) in zip(gt, pred) if p in [str(i/10.0) for i in list(range(10, 50))]]
            if predicted_rating:
                RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
                MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
            else:
                RMSE, MAE = -1, -1
            
            ## Save metrics
            with open(os.path.join(self.output_dir, task_type,'metrics.tsv'), 'a') as f:
                f.write(f"{task}\t{RMSE:.4f}\t{MAE:.4f}\n")
        
        pass

    def evaluate_task2(self):
        task_type = 'task-2'
        print("\nTesting Task 2: Sequential Recommendation..")
        os.makedirs(os.path.join(self.output_dir, task_type), exist_ok=True)
        with open(os.path.join(self.output_dir, task_type,'metrics.tsv'), 'a') as f:
                f.write(f"task\thit@5\tndcg@5\thit@10\tndcg@10\n")
        ## Load test data
        for task_num, task in enumerate(self.test_task_list['sequential']):
            print(f"{task}: {task_num+1:>2d}/{len(self.test_task_list['sequential'])}")
            test_loader = get_loader(
                args=self.args,
                task_list= {'sequential': [task]},
                sample_numbers=self.sample_numbers,
                split=self.data_type,
                mode='test',
                batch_size=self.args.batch_size,
                workers=0,
                distributed=False,
                tokenizer=self.tokenizer,
            )

            ## Evaluate
            source_text, gt, pred = [], [], []
            for i, batch in enumerate(test_loader):
                with torch.no_grad():
                    results = self.model.generate(
                        batch['input_ids'].to(self.device),
                        max_length=50,
                        num_beams=self.args.num_beams,
                        no_repeat_ngram_size=0,
                        num_return_sequences=self.args.num_beams,
                        early_stopping=True,
                    ) # batch * num_beams * max_length
                    generated = self.tokenizer.batch_decode(results, skip_special_tokens=True)
                    source_text.extend(batch['source_text'])
                    gt.extend(batch['target_text'])
                    for j in range(0, len(generated), self.args.num_beams):
                        pred.append(generated[j:j+self.args.num_beams])


            ## Save results
            self.save_results(source_text=source_text, gt=gt, pred=pred, task=task, task_type=task_type)

            from IPython import embed; embed()
            ## Calculate metrics
            ui_scores = {}
            for i, pred in enumerate(pred):
                pred_dict = {}
                for j, p in enumerate(pred):
                    pred_dict[p] = -(j+1)
                ui_scores[i] = pred_dict
            
            from IPython import embed; embed()
            metric5 = evaluate_all(ui_scores, gt, 5)[1]
            metric10 = evaluate_all(ui_scores, gt, 10)[1]

            ## Save metrics
            with open(os.path.join(self.output_dir, task_type,'metrics.tsv'), 'a') as f:
                f.write(f"{task}\t{metric5['hit']}\t{metric10['hit']}\t{metric5['ndcg']}\t{metric10['ndcg']}\n")

        return 1

    def evaluate_task3(self):
        task_type = 'task-3'
        print("\nTesting Task 3: Explanation Generation")
        os.makedirs(os.path.join(self.output_dir, task_type), exist_ok=True)
        with open(os.path.join(self.output_dir, task_type,'metrics.tsv'), 'a') as f:
            f.write(f"task\tBLEU4\trouge-1\trouge-2\trouge-l\n")
        for task_num, task in enumerate(self.test_task_list['explanation']):
            os.makedirs(os.path.join(self.output_dir, task_type, task), exist_ok=True)
            print(f"{task}: {task_num+1:>2d}/{len(self.test_task_list['explanation'])}")
            test_loader = get_loader(
                args=self.args,
                task_list= {'explanation': [task]},
                sample_numbers=self.sample_numbers,
                split=self.data_type,
                mode='test',
                batch_size=self.args.batch_size//2,
                workers=0,
                distributed=False,
                tokenizer=self.tokenizer,
            )

            ## Evaluate
            source_text, gt, pred = [], [], []
            for i, batch in enumerate(test_loader):
                with torch.no_grad():
                    results = self.model.generate(
                        batch['input_ids'].to(self.device),
                        min_length=9,
                        num_beams=12,
                        num_return_sequences=1,
                        num_beam_groups=3,
                        repetition_penalty=0.7
                    )
                    generated = self.tokenizer.batch_decode(results, skip_special_tokens=True)
                    source_text.extend(batch['source_text'])
                    gt.extend(batch['target_text'])
                    pred.extend(generated)
            
            ## Save results
            self.save_results(source_text=source_text, gt=gt, pred=pred, task=task, task_type=task_type)

            ## Calculate metrics
            new_tokens_predict = [l.split() for l in pred]
            new_tokens_test = [ll.split() for ll in gt]
            BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
            ROUGE = rouge_score(references=gt, generated=pred)

            ## Save metrics
            with open(os.path.join(self.output_dir, task_type,'metrics.tsv'), 'a') as f:
                f.write(f"{task}\t{BLEU4:.4f}\t{ROUGE['rouge_1/f_score']:.4f}\t{ROUGE['rouge_2/f_score']:.4f}\t{ROUGE['rouge_l/f_score']:.4f}\n")
        
        return

    def evaluate_task4(self):
        task_type = 'task-4'
        print("\nTesting Task 4: Review Related")
        os.makedirs(os.path.join(self.output_dir, task_type), exist_ok=True)
        with open(os.path.join(self.output_dir, task_type,'metrics.tsv'), 'a') as f:
            f.write(f"task\tBLEU2\trouge-1\trouge-2\trouge-l\n")
        for task_num, task in enumerate(self.test_task_list['review']):
            os.makedirs(os.path.join(self.output_dir, task_type, task), exist_ok=True)
            print(f"{task}: {task_num+1:>2d}/{len(self.test_task_list['review'])}")
            test_loader = get_loader(
                args=self.args,
                task_list= {'review': [task]},
                sample_numbers=self.sample_numbers,
                split=self.data_type,
                mode='test',
                batch_size=self.args.batch_size,
                workers=0,
                distributed=False,
                tokenizer=self.tokenizer,
            )

            ## Evaluate
            source_text, gt, pred = [], [], []
            for i, batch in enumerate(test_loader):
                with torch.no_grad():
                    results = self.model.generate(
                        batch['input_ids'].to(self.device),
                    )
                    generated = self.tokenizer.batch_decode(results, skip_special_tokens=True)
                    source_text.extend(batch['source_text'])
                    gt.extend(batch['target_text'])
                    pred.extend(generated)
            
            ## Save results
            self.save_results(source_text=source_text, gt=gt, pred=pred, task=task, task_type=task_type)

            ## Calculate metrics
            if task in ['4-2','4-4']:
                predicted_rating = [(float(r), round(float(p))) for (r, p) in zip(gt, pred)]
                if predicted_rating:
                    RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
                    MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
                else:
                    RMSE, MAE = -1, -1
                    
                with open(os.path.join(self.output_dir, task_type,'metrics.tsv'), 'a') as f:
                    f.write(f"{task}\t{RMSE:.4f}\t{MAE:.4f}\n")
                
            
            elif task in ['4-1', '4-3']:
                new_tokens_predict = [l.split() for l in pred]
                new_tokens_test = [ll.split() for ll in gt]
                BLEU2 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=2, smooth=False)
                ROUGE = rouge_score(references=gt, generated=pred)

                with open(os.path.join(self.output_dir, task_type,'metrics.tsv'), 'a') as f:
                    f.write(f"{task}\t{BLEU2:.4f}\t{ROUGE['rouge_1/f_score']:.4f}\t{ROUGE['rouge_2/f_score']:.4f}\t{ROUGE['rouge_l/f_score']:.4f}\n")

    def evaluate_task5(self):
        task_type = 'task-5'
        print("\nTesting Task 5: Direct Recommendation..")
        os.makedirs(os.path.join(self.output_dir, task_type), exist_ok=True)
        with open(os.path.join(self.output_dir, task_type,'metrics.tsv'), 'a') as f:
                f.write(f"task\thit@5\thit@5\tndcg@5\thit@10\tndcg@10\n")
        for task_num, task in enumerate(self.test_task_list['traditional']):
            os.makedirs(os.path.join(self.output_dir, task_type, task), exist_ok=True)
            print(f"{task}: {task_num+1:>2d}/{len(self.test_task_list['traditional'])}")
            test_loader = get_loader(
                args=self.args,
                task_list= {'traditional': [task]},
                sample_numbers=self.sample_numbers,
                split=self.data_type,
                mode='test',
                batch_size=self.args.batch_size,
                workers=0,
                distributed=False,
                tokenizer=self.tokenizer,
            )

            ## Evaluate
            source_text, gt, pred = [], [], []
            if task in ['5-1', '5-2', '5-3', '5-4']:
                for i, batch in enumerate(test_loader):
                    with torch.no_grad():
                        results = self.model.generate(
                            batch['input_ids'].to(self.device),
                        )
                        generated = self.tokenizer.batch_decode(results, skip_special_tokens=True)
                        source_text.extend(batch['source_text'])
                        gt.extend(batch['target_text'])
                        pred.extend(generated)

            elif task in ['5-5', '5-6', '5-7', '5-8']:
                for i, batch in enumerate(test_loader):
                    with torch.no_grad():
                        results = self.model.generate(
                            batch['input_ids'].to(self.device),
                            max_length=10,
                            num_beams=self.args.num_beams,
                            num_return_sequences=self.args.num_beams,
                            no_repeat_ngram_size=0,
                            early_stopping=True,
                        )
                        generated = self.tokenizer.batch_decode(results, skip_special_tokens=True)
                        source_text.extend(batch['source_text'])
                        gt.extend(batch['target_text'])
                        for j in range(0, len(generated), self.args.num_beams):
                            pred.append(generated[j:j+self.args.num_beams])
            ## Save results
            self.save_results(source_text=source_text, gt=gt, pred=pred, task=task, task_type=task_type)


            if task in ['5-5', '5-6', '5-7', '5-8']:
                ## Calculate metrics
                ui_scores = {}
                for i, pred in enumerate(pred):
                    pred_dict = {}
                    for j, p in enumerate(pred):
                        pred_dict[p] = -(j+1)
                    ui_scores[i] = pred_dict
                
                metric1 = evaluate_all(ui_scores, gt, 1)[1]
                metric5 = evaluate_all(ui_scores, gt, 5)[1]
                metric10 = evaluate_all(ui_scores, gt, 10)[1]

                ## Save metrics
                with open(os.path.join(self.output_dir, task_type,'metrics.tsv'), 'a') as f:
                    f.write(f"{task}\t{metric1['hit']}\t{metric5['hit']}\t{metric10['hit']}\t{metric5['ndcg']}\t{metric10['ndcg']}\n")

        return 1

    def save_results(self, source_text, gt, pred, task, task_type):
        total = []
        for s, g, p in zip(source_text, gt, pred):
            total.append( {
                'source_text': s,
                'gt': g,
                'pred': p
            })
        os.makedirs(os.path.join(self.output_dir, task_type, task), exist_ok=True)
        json.dump(total, open(os.path.join(self.output_dir, task_type, task, 'results.json'), 'w'), indent=4)

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

        
        config = self.create_config(args)
        model_class = P5Pretraining
        model_name = args.backbone
        model = model_class.from_pretrained(
            model_name,
            config=config
        )
        print(f"Loading Model from {model_path}..")
        model.resize_token_embeddings(self.tokenizer.vocab_size)
        state_dict = load_state_dict(model_path, self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)

        return model
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
# import file from parent directory
import sys
import os.path
import os
from typing import Optional

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
from notebooks.evaluate.utils import (rouge_score, bleu_score, unique_sentence_percent, root_mean_square_error,
                                      mean_absolute_error)


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
    with open(path, 'r') as fd:
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
        _, self.model_type, model_info, self.task_type = args.load.split('/')  # snap/naive/beauty-small-42/task-1
        self.data_type, _, self.seed = model_info.split('-')
        self.output_dir = os.path.join(project_dir, 'output', self.model_type, model_info, self.task_type)
        os.makedirs(self.output_dir, exist_ok=True)

        ## Load pretrained model and tokenizer

        if self.model_type == 'naive':
            self.tokenizer = self.create_tokenizer(tokenizer_path=None, args=args)
            self.model = self.create_model(
                    model_path=os.path.join(project_dir, args.load, 'BEST_EVAL_LOSS.pth'),
                    args=args
            )
        elif self.model_type == 'atomic':
            self.tokenizer = self.create_tokenizer(
                    tokenizer_path=os.path.join(project_dir, args.load, 'tokenizer-0/'), args=args
            )
            self.model = self.create_model(
                    model_path=os.path.join(project_dir, args.load, 'BEST_EVAL_LOSS.pth'),
                    args=args
            )
        else:
            raise NotImplementedError(f"model type {self.model_type} not implemented")

        self.model.tokenizer = self.tokenizer
        print(f"Vocab size: {len(self.tokenizer)}")

        ## Load test data  
        data_splits = load_pickle(os.path.join(project_dir, f'data/{self.data_type}/rating_splits_augmented.pkl'))
        test_review_data = data_splits['test']
        print(f"\n\nLoaded {len(test_review_data)} test reviews")

        data_maps = load_json(os.path.join(project_dir, f'data/{self.data_type}/datamaps.json'))
        print(f"number of users {len(data_maps['user2id'])}")
        print(f"number of items {len(data_maps['item2id'])}")

        if self.data_type == 'yelp':
            self.test_task_list = {'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
                                   'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10',
                                                  '2-11', '2-12'],
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
                                   'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10',
                                                  '2-11', '2-12', '2-13'],
                                   'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9',
                                                   '3-10', '3-11', '3-12'],
                                   'review': ['4-1', '4-2', '4-3', '4-4'],
                                   'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7', '5-8']}

        self.sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1,
                               'traditional': (1, 1)}

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

    # Task 1: Rating Prediction
    def evaluate_task1(self):

        task_type = 'task-1'
        print("\nTesting Task 1: Rating Prediction")
        os.makedirs(os.path.join(self.output_dir, task_type), exist_ok=True)
        with open(os.path.join(self.output_dir, task_type, 'metrics.tsv'), 'a') as f:
            f.write(f"task\tRMSE\tMAE\n")

        for task_num, task in enumerate(self.test_task_list['rating']):
            print(f"{task}: {task_num + 1:>2d}/{len(self.test_task_list['rating'])}")

            # generate
            test_loader = self.create_loader(task_name='rating', task_type=[task])
            source_text, gt, pred = self.generate_single(test_loader=test_loader)
            self.save_results(source_text=source_text, gt=gt, pred=pred, task=task, task_type=task_type)

            # binary output
            if task in ['1-3', '1-4', '1-8', '1-9']:
                continue
            # rating output
            else:
                # evaluate
                predicted_rating = []
                invalid_count = 0
                for r, p in zip(gt, pred):
                    try:
                        predicted_rating.append((float(r), float(p)))
                    except ValueError:
                        invalid_count += 1

                print(f"Invalid count: {invalid_count}")

                if predicted_rating:
                    RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
                    MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
                else:
                    RMSE, MAE = -1, -1

                with open(os.path.join(self.output_dir, task_type, 'metrics.tsv'), 'a') as f:
                    f.write(f"{task}\t{RMSE:.4f}\t{MAE:.4f}\n")

        return 1

    # Task 2: Sequential Recommendation
    def evaluate_task2(self):

        task_type = 'task-2'
        print("\nTesting Task 2: Sequential Recommendation..")
        os.makedirs(os.path.join(self.output_dir, task_type), exist_ok=True)
        with open(os.path.join(self.output_dir, task_type, 'metrics.tsv'), 'a') as f:
            f.write(f"task\thit@5\tndcg@5\thit@10\tndcg@10\n")

        for task_num, task in enumerate(self.test_task_list['sequential']):
            print(f"{task}: {task_num + 1:>2d}/{len(self.test_task_list['sequential'])}")
            test_loader = self.create_loader(task_name='sequential', task_type=[task],
                                             reduce_batch_size=task not in ['2-11', '2-12'])

            # binary output
            if task in ['2-11', '2-12']:
                # generate
                source_text, gt, pred = self.generate_single(test_loader=test_loader)
                self.save_results(source_text=source_text, gt=gt, pred=pred, task=task, task_type=task_type)

            # item output
            else:
                source_text, gt, pred = self.generate_multi_beam(test_loader=test_loader)
                self.save_results(source_text=source_text, gt=gt, pred=pred, task=task, task_type=task_type)

                # evaluate
                ui_scores = {}
                for i, pred in enumerate(pred):
                    pred_dict = {}
                    for j, p in enumerate(pred):
                        pred_dict[p] = -(j + 1)
                    ui_scores[i] = pred_dict

                metric5 = evaluate_all(ui_scores, gt, 5)[1]
                metric10 = evaluate_all(ui_scores, gt, 10)[1]

                with open(os.path.join(self.output_dir, task_type, 'metrics.tsv'), 'a') as f:
                    f.write(f"{task}\t{metric5['hit']}\t{metric10['hit']}\t{metric5['ndcg']}\t{metric10['ndcg']}\n")

        return 1

    # Task 3: Explanation Generation
    def evaluate_task3(self):

        task_type = 'task-3'
        print("\nTesting Task 3: Explanation Generation")
        os.makedirs(os.path.join(self.output_dir, task_type), exist_ok=True)
        with open(os.path.join(self.output_dir, task_type, 'metrics.tsv'), 'a') as f:
            f.write(f"task\tBLEU4\trouge-1\trouge-2\trouge-l\n")

        for task_num, task in enumerate(self.test_task_list['explanation']):
            os.makedirs(os.path.join(self.output_dir, task_type, task), exist_ok=True)
            print(f"{task}: {task_num + 1:>2d}/{len(self.test_task_list['explanation'])}")

            # generate
            test_loader = self.create_loader(task_name='explanation', task_type=[task])
            source_text, gt, pred = self.generate_single_beam(test_loader=test_loader)
            self.save_results(source_text=source_text, gt=gt, pred=pred, task=task, task_type=task_type)

            # evaluate
            new_tokens_predict = [l.split() for l in pred]
            new_tokens_test = [ll.split() for ll in gt]
            BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
            ROUGE = rouge_score(references=gt, generated=pred)

            with open(os.path.join(self.output_dir, task_type, 'metrics.tsv'), 'a') as f:
                f.write(
                        f"{task}\t{BLEU4:.4f}\t{ROUGE['rouge_1/f_score']:.4f}\t{ROUGE['rouge_2/f_score']:.4f}\t{ROUGE['rouge_l/f_score']:.4f}\n")

        return 1

    # Task 4: Review Related
    def evaluate_task4(self):

        task_type = 'task-4'
        print("\nTesting Task 4: Review Related")
        os.makedirs(os.path.join(self.output_dir, task_type), exist_ok=True)
        with open(os.path.join(self.output_dir, task_type, 'metrics.tsv'), 'a') as f:
            f.write(f"task\tBLEU2\trouge-1\trouge-2\trouge-l\n")

        for task_num, task in enumerate(self.test_task_list['review']):
            os.makedirs(os.path.join(self.output_dir, task_type, task), exist_ok=True)
            print(f"{task}: {task_num + 1:>2d}/{len(self.test_task_list['review'])}")

            # generate
            test_loader = self.create_loader(task_name='review', task_type=[task])
            source_text, gt, pred = self.generate_single(test_loader=test_loader)
            self.save_results(source_text=source_text, gt=gt, pred=pred, task=task, task_type=task_type)

            # evaluate
            # rating output
            if task in ['4-2', '4-4']:
                predicted_rating = []
                invalid_count = 0
                for r, p in zip(gt, pred):
                    try:
                        predicted_rating.append((float(r), float(p)))
                    except ValueError:
                        invalid_count += 1

                print(f"Invalid count: {invalid_count}")

                if predicted_rating:
                    RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
                    MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
                else:
                    RMSE, MAE = -1, -1

                with open(os.path.join(self.output_dir, task_type, 'metrics.tsv'), 'a') as f:
                    f.write(f"{task}\t{RMSE:.4f}\t{MAE:.4f}\n")

            # summary output
            elif task in ['4-1', '4-3']:
                new_tokens_predict = [l.split() for l in pred]
                new_tokens_test = [ll.split() for ll in gt]
                BLEU2 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=2, smooth=False)
                ROUGE = rouge_score(references=gt, generated=pred)

                with open(os.path.join(self.output_dir, task_type, 'metrics.tsv'), 'a') as f:
                    f.write(
                            f"{task}\t{BLEU2:.4f}\t{ROUGE['rouge_1/f_score']:.4f}\t{ROUGE['rouge_2/f_score']:.4f}\t{ROUGE['rouge_l/f_score']:.4f}\n")

        return 1

    # Task 5: Direct Recommendation
    def evaluate_task5(self):

        task_type = 'task-5'
        print("\nTesting Task 5: Direct Recommendation..")
        os.makedirs(os.path.join(self.output_dir, task_type), exist_ok=True)
        with open(os.path.join(self.output_dir, task_type, 'metrics.tsv'), 'a') as f:
            f.write(f"task\thit@5\thit@5\tndcg@5\thit@10\tndcg@10\n")

        for task_num, task in enumerate(self.test_task_list['traditional']):
            os.makedirs(os.path.join(self.output_dir, task_type, task), exist_ok=True)
            print(f"{task}: {task_num + 1:>2d}/{len(self.test_task_list['traditional'])}")

            # generate
            test_loader = self.create_loader(task_name='traditional', task_type=[task])
            if task in ['5-1', '5-2', '5-3', '5-4']:  # binary output
                source_text, gt, pred = self.generate_single(test_loader=test_loader)
                self.save_results(source_text=source_text, gt=gt, pred=pred, task=task, task_type=task_type)
            elif task in ['5-5', '5-6', '5-7', '5-8']:  # item output
                source_text, gt, pred = self.generate_topk(test_loader=test_loader)
                self.save_results(source_text=source_text, gt=gt, pred=pred, task=task, task_type=task_type)

                # evaluate
                if task in ['5-5', '5-6', '5-7', '5-8']:
                    ui_scores = {}
                    for i, pred in enumerate(pred):
                        pred_dict = {}
                        for j, p in enumerate(pred):
                            pred_dict[p] = -(j + 1)
                        ui_scores[i] = pred_dict

                    metric1 = evaluate_all(ui_scores, gt, 1)[1]
                    metric5 = evaluate_all(ui_scores, gt, 5)[1]
                    metric10 = evaluate_all(ui_scores, gt, 10)[1]

                    with open(os.path.join(self.output_dir, task_type, 'metrics.tsv'), 'a') as f:
                        f.write(
                                f"{task}\t{metric1['hit']}\t{metric5['hit']}\t{metric10['hit']}\t{metric5['ndcg']}\t{metric10['ndcg']}\n")

        return 1

    '''
    Helper functions
    '''

    def create_loader(self, task_name: str, task_type: list, reduce_batch_size: bool = False):
        '''
        Create test loader for a specific task
        task_name: str, name of the task (e.g. review, traditional)
        task_type: list, list of task types (e.g. ['4-1', '4-2'])
        reduce_batch_size: bool, whether to reduce batch size to 1/4th of the original batch size
        '''
        assert task_name in self.test_task_list.keys(), f"Task name {task_name} not found in test task list"

        batch_size = self.args.batch_size
        if reduce_batch_size:
            batch_size = int(batch_size / 16)

        return get_loader(
                args=self.args,
                task_list={task_name: task_type},
                sample_numbers=self.sample_numbers,
                split=self.data_type,
                mode='test',
                batch_size=batch_size,
                workers=4,
                distributed=False,
                tokenizer=self.tokenizer,
        )

    def generate_single(self, test_loader):
        '''
        Generate single output with greedy search
        '''
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
        return source_text, gt, pred

    def generate_single_beam(self, test_loader):
        '''
        Generate single output with beam search
        '''
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
        return source_text, gt, pred

    def generate_multi_beam(self, test_loader):
        '''
        Generate multiple outputs with beam search
        '''
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
                )  # batch * num_beams * max_length
                generated = self.tokenizer.batch_decode(results, skip_special_tokens=True)
                source_text.extend(batch['source_text'])
                gt.extend(batch['target_text'])
                for j in range(0, len(generated), self.args.num_beams):
                    pred.append(generated[j:j + self.args.num_beams])
        return source_text, gt, pred

    def save_results(self, source_text, gt, pred, task, task_type):
        '''
        Save results to json file
        source_text: list, list of source text
        gt: list, list of ground truth
        pred: list, list of predictions
        task: str, name of the task (e.g. task-1, task-2)
        task_type: str, name of the task type (e.g. 1-1, 1-2)
        '''
        total = []
        for s, g, p in zip(source_text, gt, pred):
            total.append({
                'source_text': s,
                'gt': g,
                'pred': p
            })
        os.makedirs(os.path.join(self.output_dir, task_type, task), exist_ok=True)
        json.dump(total, open(os.path.join(self.output_dir, task_type, task, 'results.json'), 'w'), indent=4)

    @staticmethod
    def create_tokenizer(tokenizer_path: Optional[str], args):
        # Workaround for loading tokenizer taking too long
        tokenizer = T5TokenizerFast.from_pretrained(args.backbone)

        if tokenizer_path is not None:
            print(f"Loading Tokenizer from {tokenizer_path}..")

            added_token = json.load(open(os.path.join(tokenizer_path, "added_tokens.json")))
            # Sort by token ID
            added_token = [x[0] for x in sorted(added_token.items(), key=lambda y: y[1])]

            print(f"Adding {len(added_token)} extra tokens..")

            tokenizer.add_tokens(added_token, special_tokens=True)

        else:
            print("No extra tokens.")

        return tokenizer

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
        model.resize_token_embeddings(len(self.tokenizer))
        print(f"Resized Token Embeddings to {len(self.tokenizer)}...")
        state_dict = load_state_dict(model_path, "cpu")
        model.load_state_dict(state_dict)
        model.to(self.device)

        return model


if __name__ == "__main__":
    args = parse_args()
    main(args)

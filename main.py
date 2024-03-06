import torch.nn as nn
from transformers import AutoTokenizer
import os
import json
from tqdm import tqdm
import base_config as config
from router.modeling import Router
from router.train import train_router
from prepare_data.config import categories
from prepare_data.main import prepare_data
from innermoe.modeling import InnerMoERM
from innermoe.train_pipe import train_pipe


class DMoERM(nn.Module):

    def __init__(self):
        super(DMoERM, self).__init__()


    def train(self):

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)

        self.router = Router().to(config.device)

        # train the router
        print('-' * 15 + 'begin training router' + '-' * 15)
        train_router(self.router)

        # obtain preference labels on each capability point, it will be used in training Phase 2
        print('-' * 15 + 'begin acquiring capability point preference labels' + '-' * 15)
        prepare_data()

        # train each innerMoERM
        self.innerMoERM_list = []
        for cat in categories:
            print('-' * 15 + 'begin training InnerMoE-' + cat + '-' * 15)
            innerMoERM = InnerMoERM().to(config.device)
            self.innerMoERM_list.append(innerMoERM)
            train_pipe(innerMoERM, cat)
            innerMoERM.to('cpu')  # move trained innerMoERM to cpu temprarily, for training needs more gpu memory

        # move trained innerMoERM back to gpu
        for innerMoERM in self.innerMoERM_list:
            innerMoERM.to(config.device)

        print('-' * 15 + 'finish training' + '-' * 15)


    def test(self):

        print('-' * 15 + 'begin testing' + '-' * 15)

        with open(os.path.join(config.data_dir, 'Ernie-rlhf-test.jsonl'), 'r') as f:
            lines = [json.loads(line) for line in f.readlines()]
        
        results = {}
        results['all'] = {'acc': 0, 'tot': 0}

        # test the consistency with human annonation
        for line in tqdm(lines[:20]):

            if line['label'] not in results.keys():
                results[line['label']] = {'acc': 0, 'tot': 0}
            
            rewards = []
            for response in line['response']:
                context = {'src': line['src'], 'tgt': line['tgt'] + [response]}
                reward = self.forward(context)
                rewards.append(reward)

            for i, rank1 in enumerate(line['rank']):
                for j, rank2 in enumerate(line['rank'][i + 1:]):
                    if rank1 > rank2 and rewards[i] > rewards[j] \
                        or rank1 < rank2 and rewards[i] < rewards[j]:
                        results[line['label']]['acc'] += 1
                        results['all']['acc'] += 1
                    if rank1 != rank2:
                        results[line['label']]['tot'] += 1
                        results['all']['tot'] += 1

        for key in results.keys():
            results[key]['acc_rate'] = results[key]['acc'] / results[key]['tot']
        
        print('-' * 15 + 'finish testing' + '-' * 15)
        print(results)


    def forward(self, context):

        # route the input to the corresponding category
        cat = self.router.route(context, self.tokenizer)

        # use the corresponding innerMoERM to get reward
        final_reward = self.innerMoERM_list[cat].get_reward(context, self.tokenizer)
        
        return final_reward


if __name__ == '__main__':
    model = DMoERM()
    model.train()
    model.test()

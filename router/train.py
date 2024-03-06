import torch.nn as nn
import torch
from transformers import AutoTokenizer
import torch.optim as optim
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import router.config as config


def train_router(router):

    if os.path.exists(os.path.join(config.out_dir, 'best_router.pth')):
        print('already trained router, passed!')
        router = torch.load(os.path.join(config.out_dir, 'best_router.pth'))
        return

    with open(os.path.join(config.rawdata_dir, 'Ernie-rlhf-train.jsonl'), 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)

    user_token = '<extra_0>'
    dataset = []
    for line in tqdm(lines):
        src = ''
        for query in line['src']:
            src += user_token + query
        tokenized = tokenizer(src, return_tensors="pt")
        for index, cat in enumerate(config.cat_list):
            if cat == line['label']:
                label = torch.tensor([0] * len(config.cat_list))
                label[index] = 1
                dataset.append({
                    'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized['attention_mask'],
                    'label': label.double()
                })
                break
        
    train_test_split = int(len(dataset) * config.val_rate)
    train_dataset, test_dataset = dataset[train_test_split:], dataset[:train_test_split]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(router.parameters(), lr=config.lr)

    step = 0
    accs = {
        'train': [],
        'val': []
    }
    steps = []
    best_acc = 0
    no_adding = 0

    router.train()
    while True:
        for train_data in tqdm(train_dataset):
            if step % config.steps_per_eval == 0:
                
                train_acc = eval_router(router, train_dataset)
                val_acc = eval_router(router, test_dataset)
                accs['train'].append(train_acc)
                accs['val'].append(val_acc)
                steps.append(step)
                
                plt.figure()
                for key, acc in accs.items():
                    plt.plot(steps, acc, label=key)
                plt.title('Router')
                plt.xlabel('step')
                plt.ylabel('acc')
                plt.legend(loc="best")
                plt.savefig(os.path.join(config.out_dir, 'testing_acc_router.png'))
                plt.close()

                if val_acc > best_acc:
                    best_acc = val_acc
                    no_adding = 0
                    torch.save(router, os.path.join(config.out_dir, 'best_router.pth'))
                else:
                    no_adding += 1
                    if no_adding >= config.max_no_adding_times:
                        print(f'finish training! best router acc: {best_acc}')
                        route_dataset(router)
                        return
            
            train_data['label'] = train_data['label'].to(config.device)
            train_data['input_ids'] = train_data['input_ids'].to(config.device)
            train_data['attention_mask'] = train_data['attention_mask'].to(config.device)
            output = router(train_data['input_ids'], train_data['attention_mask'])
            
            loss = criterion(output, train_data['label'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1


def eval_router(router, tokenized_dataset):
    router.eval()
    tot = 0
    acc = 0
    with torch.no_grad():

        print(f'begin testing!')
        lenth = len(tokenized_dataset)
        random_numbers = random.sample(range(lenth), lenth)
        for index in tqdm(random_numbers):

            tokenized_data = tokenized_dataset[index]
            tokenized_data['label'] = tokenized_data['label'].to(config.device)
            tokenized_data['input_ids'] = tokenized_data['input_ids'].to(config.device)
            tokenized_data['attention_mask'] = tokenized_data['attention_mask'].to(config.device)

            output = router(tokenized_data['input_ids'], tokenized_data['attention_mask'])
            _, predicted = torch.max(output, 0)
            _, label = torch.max(tokenized_data['label'], 0)

            if predicted == label:
                acc += 1
            tot += 1
            if tot >= config.eval_samples:
                return acc / tot

    return acc / tot


def route_dataset(router):
    """
    After training, obtain the category labels for unlabeled data, which will then be used for training the corresponding innerMoERM
    """

    with open(os.path.join(config.rawdata_dir, 'Ernie-rlhf-train.jsonl'), 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)

    router.eval()
    with torch.no_grad():
        user_token = '<extra_0>'
        for line in tqdm(lines):
            if line['label'] in config.cat_list:
                continue
            src = ''
            for query in line['src']:
                src += user_token + query

            tokenized = tokenizer(src, return_tensors="pt").to(config.device)
            
            output = router(tokenized['input_ids'], tokenized['attention_mask'])
            _, predicted = torch.max(output, 0)

            line['label'] = config.cat_list[predicted]

    with open(os.path.join(config.phasedata_dir, 'routed.jsonl'), 'w') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

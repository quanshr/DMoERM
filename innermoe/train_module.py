import torch
from tqdm import tqdm
from torch import nn, optim
import matplotlib.pyplot as plt
import os
import random
import innermoe.config as config


class PairWiseLoss(nn.Module):

    def __init__(self, loss_type='logsigmoid'):
        super(PairWiseLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, w, l):
        diff = w - l
        if self.loss_type == 'direct':
            loss = -diff
        elif self.loss_type == 'logsigmoid':  # we use logsigmoid loss as default
            loss = -torch.log(torch.sigmoid(diff))
        elif self.loss_type == 'score':
            out = w
            score = l
            loss = (torch.sigmoid(out) - score / 2) ** 2
        else:
            raise Exception('Unknown loss type')
        return loss


def train_rm(innerMoERM, dataset, lr, mode, out_dir):
    """
    The interface of training inner DMoERM for three training phases
    """

    split = int(config.val_rate * len(dataset))
    train_dataset, val_dataset = dataset[:split], dataset[split:]  # split the dataset into train and val
    
    # only train the parameters that require gradient--it is important because we train different parts of the model at different training phases
    trainable = [p for p in innerMoERM.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=lr)
    criterion = PairWiseLoss()
    
    innerMoERM.train()
    step = 0
    accs = {
        'train': [],
        'val': []
    }
    steps = []
    best_val_acc = 0
    no_adding = 0
    epoch = 0
    while True:
        print(f'begin training! epoch: {epoch}')
        epoch += 1
        for tokenized_data in tqdm(train_dataset):
            if step % config.steps_per_eval == 0:
                train_acc = eval_rm(innerMoERM, train_dataset)
                val_acc = eval_rm(innerMoERM, val_dataset)
                accs['train'].append(train_acc)
                accs['val'].append(val_acc)
                steps.append(step)

                # draw the accuracy curve
                plt.figure()
                for key, acc in accs.items():
                    plt.plot(steps, acc, label=key)
                plt.xlabel('step')
                plt.ylabel('acc')
                plt.legend(loc="best")
                plt.savefig(os.path.join(out_dir, 'pictures', mode + '.png'))
                plt.close()

                if val_acc > best_val_acc:  # save the best model--we will use it for the next training phase or final testing if it is already Phase 3
                    best_val_acc = val_acc
                    no_adding = 0
                    torch.save(innerMoERM.state_dict(), os.path.join(out_dir, 'ckpts', mode + '.pth'))
                else:
                    no_adding += 1
                    if no_adding >= config.max_no_adding_times:  # early stopping if no improvement for a certain number of times
                        return best_val_acc
                    
            rewards = []
            input_ids_list = tokenized_data['input_ids']
            attention_mask_list = tokenized_data['attention_mask']
            rank = tokenized_data['rank']
            for index, input_ids in enumerate(input_ids_list):  # get the reward from model for each response
                input_ids = input_ids.to(config.device)
                attention_mask = attention_mask_list[index].to(config.device)
                reward = innerMoERM(input_ids, attention_mask)
                rewards.append(reward)
            bs = 0
            loss = 0
            for ith, rank1 in enumerate(rank):
                for jth, rank2 in enumerate(rank):
                    if ith >= jth or rank1 == rank2:
                        continue
                    bs += 1
                    reward1 = rewards[ith]
                    reward2 = rewards[jth]
                    if rank1 > rank2:
                        reward1, reward2 = reward2, reward1
                    loss += criterion(reward1, reward2)
            if bs != 0:  # optimize a step
                loss /= bs
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1


def eval_rm(innerMoERM, tokenized_dataset):
    """
    Test the model performance during training
    Return the consistency(accuracy) with human annotation
    """

    innerMoERM.eval()
    tot = 0
    acc = 0
    with torch.no_grad():
        print(f'begin testing!')
        lenth = len(tokenized_dataset)
        random_numbers = random.sample(range(lenth), lenth)  # random shuffling
        for index in tqdm(random_numbers):
            tokenized_data = tokenized_dataset[index]
            input_ids_list = tokenized_data['input_ids']
            attention_mask_list = tokenized_data['attention_mask']
            rank = tokenized_data['rank']
            
            rewards = []
            for index, input_ids in enumerate(input_ids_list):  # get the reward from model for each response
                input_ids = input_ids.to(config.device)
                attention_mask = attention_mask_list[index].to(config.device)
                reward = innerMoERM(input_ids, attention_mask)
                rewards.append(reward)
            
            for ith, rank1 in enumerate(rank):
                for jth, rank2 in enumerate(rank):
                    if ith >= jth or rank1 == rank2:
                        continue
                    tot += 1
                    reward1 = rewards[ith]
                    reward2 = rewards[jth]
                    if rank1 > rank2 and reward1 > reward2 or rank1 < rank2 and reward1 < reward2:  # consistent with human annotation
                        acc += 1
                    if tot >= config.eval_samples:  # stop when enough samples are tested
                        return acc / tot

    return acc / tot

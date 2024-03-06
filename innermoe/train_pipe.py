import torch
import os
import argparse
import innermoe.config as config
from innermoe.modeling import InnerMoERM
from innermoe.load_data import load_data
from innermoe.train_module import train_rm


def phase_train(innerMoERM, data, phase, mode, cat):
    """
    Train the model in a given training phase
    """

    print('-' * 10 + f'begin training {mode}' + '-' * 10)

    lr = config.phase_lrs[phase - 1]
    out_dir = os.path.join(config.out_dir, cat)

    ckpt_path = os.path.join(out_dir, 'ckpts', mode + '.pth')
    if os.path.exists(ckpt_path):
        print(f'already trained {mode}, passed!')
        innerMoERM.load_state_dict(torch.load(ckpt_path, map_location=config.device))
        return
    
    acc = train_rm(innerMoERM, data, lr, mode, out_dir)
    innerMoERM.load_state_dict(torch.load(ckpt_path, map_location=config.device))
    innerMoERM = innerMoERM.to(config.device)
    print(f'finish training {mode}! best acc: {acc}')


def train_pipe(innerMoERM, cat):
    """
    For a given category, train the inner DMoERM in three training phases sequentially
    """

    os.makedirs(os.path.join(config.out_dir, cat), exist_ok=True)
    os.makedirs(os.path.join(config.out_dir, cat, 'ckpts'), exist_ok=True)
    os.makedirs(os.path.join(config.out_dir, cat, 'pictures'), exist_ok=True)
    
    phase1_data, phase2_data, phase3_data = load_data(os.path.join(config.phasedata_dir, cat))
    
    phase_train(innerMoERM, phase1_data, 1, 'phase1', cat)

    innerMoERM.change_phase(2)
    for key, value in phase2_data.items():
        innerMoERM.change_cap(key)
        phase_train(innerMoERM, value, 2, f"phase2_{key}", cat)

    innerMoERM.change_phase(3)
    phase_train(innerMoERM, phase3_data, 3, 'phase3', cat)


if __name__ == '__main__':
    """
    The inner DMoERM can be trained and used independently
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--cat', type=str, required=True)
    args = parser.parse_args()
    cat = args.cat
    innerMoERM = InnerMoERM().to(config.device)
    train_pipe(innerMoERM, cat)

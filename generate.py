import sys
import os
import click

import shutil
import logging
#import coloredlogs

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from utils import load_dataset, iterate_minibatches
from model import CharRNN

SOS_TOKEN = '~'
PAD_TOKEN = '#'


def load_ckp(checkpoint_fpath, model, optimiser):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimiser: optimiser we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimiser from checkpoint to optimizer
    optimiser.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimiser, epoch value, min validation loss 
    return model, optimiser, checkpoint['epoch'], valid_loss_min #.item()

def generate_sample(model, token_to_idx, idx_to_token, max_length=20, n_tokens=20, seed_phrase=SOS_TOKEN):
    """ Generates samples using seed phrase.

    Args:
        model (nn.Module): the character-level RNN model to use for sampling.
        token_to_idx (dict of `str`: `int`): character to token_id mapping dictionary (vocab).
        idx_to_token (list of `str`): index (token_id) to character mapping list (vocab).
        max_length (int): max length of a sequence to sample using model.
        seed_phrase (str): the initial seed characters to feed the model. If unspecified, defaults to `SOS_TOKEN`.
    
    Returns:
        str: generated sample from the model using the seed_phrase.
    """
    print('generate_sample', file=sys.stderr)
    model.eval()
    if seed_phrase[0] != SOS_TOKEN:
        seed_phrase = SOS_TOKEN + seed_phrase.lower()
    try:
        # convert to token ids for model
        sequence = [token_to_idx[token] for token in seed_phrase]
    except KeyError as e:
        logging.error('unknown token: {}'.format(e))
        exit(0)
    input_tensor = torch.LongTensor([sequence])

    hidden = model.initHidden(1)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        if type(hidden) == tuple:
            hidden = tuple([x.cuda() for x in hidden])
        else:
            hidden = hidden.cuda()

    # feed the seed phrase to manipulate rnn hidden states
    for t in range(len(sequence) - 1):
        _, hidden = model(input_tensor[:, t], hidden)
    
    # start generating
    for _ in range(max_length - len(seed_phrase)):
        # sample char from previous time-step
        input_tensor = torch.LongTensor([sequence[-1]])
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        probs, hidden = model(input_tensor, hidden)

        # need to use `exp` as output is `LogSoftmax`
        probs = list(np.exp(np.array(probs.data[0].cpu())))
        # normalize probabilities to ensure sum = 1
        probs /= sum(probs)
        # sample char randomly based on probabilities
        sequence.append(np.random.choice(len(idx_to_token), p=probs))
    # format the string to ignore `pad_token` and `start_token` and return
    res = str(''.join([idx_to_token[ix] for ix in sequence 
                if idx_to_token[ix] != PAD_TOKEN and idx_to_token[ix] != SOS_TOKEN]))
    print(res, file=sys.stderr)
    return res

def main():
    inputs, token_to_idx, idx_to_token = load_dataset(file_name=sys.argv[2])

    #coloredlogs.install(level='DEBUG')
    num_layers = 2
    rnn_type='lstm'
    dropout=0.5
    emb_size = 50
    hidden_size = 256
    learning_rate = 0.001
    n_tokens = len(idx_to_token)

    model = CharRNN(num_layers=num_layers, rnn_type=rnn_type,
                    dropout=dropout, n_tokens=n_tokens,
                    emb_size=emb_size, hidden_size=hidden_size,
                    pad_id=token_to_idx[PAD_TOKEN])
    if torch.cuda.is_available():
        model = model.cuda()

    optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    try:
        model, optimiser, epoch, valid_loss_min = load_ckp(checkpoint_fpath=sys.argv[1], model=model, optimiser=optimiser)
        generate_sample(model, token_to_idx, idx_to_token, n_tokens=20)
    except KeyboardInterrupt:
        print('Aborted!')

if __name__ == '__main__':
    main()

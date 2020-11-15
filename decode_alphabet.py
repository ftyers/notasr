import copy, random
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

def score(model, token_to_idx, idx_to_token, seed_phrase):
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

    max_score = 0
    best_op = ""
    best_mapping_t2i = {}#cp_token_to_idx
    best_mapping_i2t = {}#cp_idx_to_token

    # with only |V| of 3 we probably don't need to permute 100 times :D 
    for j in range(0, 100):
        print(j, '---------------------------------------')
        cp_idx_to_token = copy.copy(idx_to_token)
        cp_idx_to_token.remove('~')
        cp_idx_to_token.remove('#')
        random.shuffle(cp_idx_to_token)
        cp_idx_to_token = ['~'] + cp_idx_to_token + ['#'] 
        cp_token_to_idx = {token: cp_idx_to_token.index(token) for token in cp_idx_to_token}
        print('cp_idx_to_token:', cp_idx_to_token, '||| idx_to_token:', idx_to_token) 
        print('cp_token_to_idx:', cp_token_to_idx, '||| token_to_idx:', token_to_idx) 
    
        model.eval()
        if seed_phrase[0] != SOS_TOKEN:
            seed_phrase = SOS_TOKEN + seed_phrase.lower()
        try:
            # convert to token ids for model
            sequence = [cp_token_to_idx[token] for token in seed_phrase]
        except KeyError as e:
            logging.error('unknown token: {}'.format(e))
            exit(0)
    
    
        print('score:', seed_phrase, file=sys.stderr)
        print('sequence:', sequence, file=sys.stderr)
        input_tensor = torch.LongTensor([sequence])
        print('input_tensor:', input_tensor, file=sys.stderr)
    
        hidden = model.initHidden(1)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            if type(hidden) == tuple:
                hidden = tuple([x.cuda() for x in hidden])
            else:
                hidden = hidden.cuda()
    
        # feed the seed phrase to manipulate rnn hidden states
#        for t in range(len(sequence) - 1):
#            _, hidden = model(input_tensor[:, t], hidden)
    
           
        # start generating
        score = 0.0
        op = ""
        for i in range(0, len(sequence)):
            # sample char from previous time-step
            #input_tensor = torch.LongTensor([sequence[-1]])
            input_tensor = torch.LongTensor([sequence[i-1]])
            #if torch.cuda.is_available():
            #    input_tensor = input_tensor.cuda()
            probs, hidden = model(input_tensor, hidden)
            # need to use `exp` as output is `LogSoftmax`
            probs = list(np.exp(np.array(probs.data[0].cpu())))
            # normalize probabilities to ensure sum = 1
            probs /= sum(probs)
            c = cp_idx_to_token[sequence[i]]
            c2 = idx_to_token[sequence[i]]
            ci = cp_token_to_idx[c]
            # should this be + or should it be * ?
            score += probs[ci]
            print(c2,'|||', c,'|||', sequence[i], probs[ci], input_tensor,  file=sys.stderr)
            op += c2
            
            # sample char randomly based on probabilities
    #        sequence.append(np.random.choice(len(idx_to_token), p=probs))
    #        sequence.append(
        # format the string to ignore `pad_token` and `start_token` and return
    #    res = str(''.join([idx_to_token[ix] for ix in sequence 
    #                if idx_to_token[ix] != PAD_TOKEN and idx_to_token[ix] != SOS_TOKEN]))
        
        print('op:', op)
        if score > max_score:
            print('MAX')
            best_op = op
            max_score = score
            best_mapping_t2i = copy.copy(cp_token_to_idx)
            best_mapping_i2t = copy.copy(cp_idx_to_token)

        print('SCORE:', score, file=sys.stderr)
        print('----------------------------------------------')

    print('best:') 
    print(best_mapping_t2i, '|||', token_to_idx)
    print(best_mapping_i2t, '|||', idx_to_token)
    print(max_score)
    print(best_op)

    return max_score

def main():

    logging.root.setLevel(logging.NOTSET)

    inputs, token_to_idx, idx_to_token = load_dataset(file_name=sys.argv[2])

    idx_to_token.remove('~')
    idx_to_token.remove('#')
    idx_to_token = ['~'] + idx_to_token + ['#'] 
    token_to_idx = {token: idx_to_token.index(token) for token in idx_to_token}

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

    s4 = "bacc bac bacc bababac bababa"

    try:
        model, optimiser, epoch, valid_loss_min = load_ckp(checkpoint_fpath=sys.argv[1], model=model, optimiser=optimiser)
        score(model, token_to_idx, idx_to_token, seed_phrase=s4)
    except KeyboardInterrupt:
        print('Aborted!')

if __name__ == '__main__':
    main()

import random, re
import numpy as np, numpy.random
from pprint import pprint
from math import log

# Creates a 'State' class for each state of the HMM.
# Can be configured to any number of states the user desires.
# Each state takes in one argument: `letter_list`,
# which stores a simple list of all "lettters" in the input text.
# Each state is to be named later based on the number of num_states
# input by the user.
class State():
    name = ''

    def __init__(self, letter_list):
        self.letter_list = letter_list

    def __repr__(self):
        return self.name

# This function takes in a list `items` as its only argument
# and returns a dictionary that randomly assigns
def probs(items):
    item_probs = {}

    probability_array = np.random.dirichlet(np.ones(len(items)), size = 1)
    combined_items_probs = list(zip(items, probability_array[0]))
    for (it, prob) in combined_items_probs:
        item_probs[it] = prob

    return item_probs

def forward(states, pi, this_word):
    alpha = {}
    for s in states:
        alpha[(s, 0)] = pi[int(s.name[-1])]
    for t in range(1, len(this_word) + 1):
        for to_state in states:
            alpha[(to_state, t)] = 0
            for from_state in states:
                alpha[(to_state, t)] += alpha[(from_state, t - 1)] * from_state.letter_probs[this_word[t - 1]] * from_state.a
    return alpha

def backward(states, this_word):
    beta = {}
    last = len(this_word)
    for s in states:
        beta[(s, last)] = 1
    for t in range(len(this_word) - 1, -1, -1):
        for from_state in states:
            beta[(from_state, t)] = 0
            for to_state in states:
                beta[(from_state, t)] += beta[(to_state, t + 1)] * from_state.letter_probs[this_word[t]] * from_state.a
    return beta

def get_plog(a):
    return -log(a, 2)

letters = []
wds = []

with open('english1000.txt') as f:
    for line in f:
        line = line[:-1] + '#'
        line = line.lower()
        wds.append(line)
        for c in line:
            if c not in letters and re.search('[a-z]|#|\'|\.', c):
                letters.append(c)

num_states = int(input('How many states would you like to use?: '))
VerboseFlag = False

states = []
pis = list(range(num_states))
pi_probs = probs(pis)

for num in range(num_states):
    states.append(State(letters))

m = 0
for state in states:
    state.name = f'State {str(m)}'
    state.a = round(1 / num_states, 4)
    state.letter_probs = probs(state.letter_list)
    m += 1

alphas = {}
betas = {}
alpha_totals = {}
beta_totals = {}
plogs = {}

for w in wds:
    alphas[w] = forward(states, pi_probs, w)
    betas[w] = backward(states, w)
    alpha_totals[w] = 0
    beta_totals[w] = 0

for w in alphas:
    for alph in alphas[w]:
        s, t = alph
        if t == len(alphas[w]) / 2 - 1: # This is a problem for odd number of states
            alpha_totals[w] += alphas[w][alph]

for w in betas:
    for bet in betas[w]:
        s, t = bet
        if t == 0:
            beta_totals[w] += pi_probs[int(s.name[-1])] * betas[w][bet]

for alph_sum in alpha_totals:
    plog = get_plog(alpha_totals[alph_sum])
    plogs[alph_sum] = plog

#pprint(plogs)
#pprint(sum(plogs.values()))

pprint(alpha_totals['yeah#'])
pprint(beta_totals['yeah#'])

j = 0
if VerboseFlag == True:
    print('-' * 33, '\n')
    print('-\t Initialization\t\t -')
    print('-' * 33)

    for state in states:
        print(f'\nCreating State {j}')
        print('Transitions')

        for num in range(num_states):
            print(f'\tTo state\t{num}\t{state.a}')

        print('\nEmission probabilities')
        for letter in state.letter_probs:
            print(f'\tLetter\t{letter}\t{state.letter_probs[letter]}')

        j += 1

    print('\n', '-' * 33)
    print('Pi:')
    for pi in pi_probs:
        print(f'State\t{pi}\t{pi_probs[pi]}')

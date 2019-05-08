import random, re
from pprint import pprint
from math import log

class State():
    name = ''

    def __init__(self, letter_list):
        self.letter_list = letter_list

    def __repr__(self):
        return self.name

def probs(items):
    random.shuffle(items)
    item_probs = {}

    sum = 1
    i = 0
    for item in items:
        if i < len(items) - 1:
            rand = random.uniform(0, sum)
            item_probs[item] = rand
            sum -= rand
        else:
            item_probs[item] = sum
        i += 1

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

def soft_counts(states, this_word, alphas, betas, totals):
    sc = {}
    for t in range(0, len(this_word)):
        for from_state in states:
            for to_state in states:
                sc[(this_word[t], from_state, to_state)] = (alphas[this_word][(from_state, t)] * from_state.a * from_state.letter_probs[this_word[t]] * betas[this_word][(to_state, t + 1)]) / totals[this_word]
    return sc

def get_plog(x):
    return -log(x, 2)

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

word_soft_counts = {}

for w in wds:
    word_soft_counts[w] = soft_counts(states, w, alphas, betas, alpha_totals)

pprint(word_soft_counts)

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
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
# and returns a dictionary that randomly assigns values
# between 0 and 1 via the Dirichlet distribution.
def probs(items):
    item_probs = {}

    probability_array = np.random.dirichlet(np.ones(len(items)), size = 1)
    combined_items_probs = list(zip(items, probability_array[0]))
    for (it, prob) in combined_items_probs:
        item_probs[it] = prob

    return item_probs

# This function calculates the forward variable `alpha`
# for each letter (`t`) for each state specified.
# This function was originally provided by Prof. Goldsmith in a handout
# and adjusted slightly to cooperate with the rest of my program.
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

# This function calculates the backward variable `beta`
# for each letter (`t`) for each state specified.
# This function was originally provided by Prof. Goldsmith in a handout
# and adjusted slightly to cooperate with the rest of my program.
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

# This function calculates the soft counts for each letter of the input word
# for each state transition possible and returns those values in a dictionary.
def soft_counts(states, this_word, alphas, betas, totals):
    sc = {}
    for t in range(0, len(this_word)):
        for from_state in states:
            for to_state in states:
                sc[(this_word[t], from_state, to_state)] = (alphas[this_word][(from_state, t)] * from_state.a * from_state.letter_probs[this_word[t]] * betas[this_word][(to_state, t + 1)]) / totals[this_word]
    return sc

# Calculates the inverse log base 2 of a number `x`.
# Prof. Goldsmith refers to this as the "plog"---hence the function name.
def get_plog(x):
    return -log(x, 2)

letters = []
wds = []

# Reads file for analysis.
# Assumes that file contains one word per line.
with open('english1000.txt') as f:
    for line in f:
        line = line[:-1] + '#'
        line = line.lower()
        wds.append(line)
        for c in line:
            if c not in letters and re.search('[a-z]|#|\'|\.', c):
                letters.append(c)

num_states = int(input('How many states would you like to use?: '))

states = []
pis = list(range(num_states))
pi_probs = probs(pis)

# Create each state and add it to the list of states (`states`).
for num in range(num_states):
    states.append(State(letters))

# Creates properties for each state.
# Assigns an even transition probability (`state.a`) initially; this
# will be updated in a later iteration of the project.
m = 0
for state in states:
    state.name = f'State {str(m)}'
    state.a = round(1 / num_states, 4)
    state.letter_probs = probs(state.letter_list)
    m += 1

alphas = {}
betas = {}
alpha_totals = {} # String probability for each word given its alphas.
beta_totals = {} # String probability for each word given its betas.
plogs = {}

# Calculates the alphas and betas for each letter for each
# possible state transition.
# Also initializes string probability for each word given the
# alphas and betas.
for w in wds:
    alphas[w] = forward(states, pi_probs, w)
    betas[w] = backward(states, w)
    alpha_totals[w] = 0
    beta_totals[w] = 0

# Calculates string probability given its alphas.
for w in alphas:
    for alph in alphas[w]:
        s, t = alph
        if t == len(alphas[w]) / 2 - 1: # This is a problem for odd number of states
            alpha_totals[w] += alphas[w][alph]

# Calculates string probability given its betas.
for w in betas:
    for bet in betas[w]:
        s, t = bet
        if t == 0:
            beta_totals[w] += pi_probs[int(s.name[-1])] * betas[w][bet]

# Calculates the plog of each word.
for alph_sum in alpha_totals:
    plog = get_plog(alpha_totals[alph_sum])
    plogs[alph_sum] = plog

letter_soft_counts = {}

# Calculates the soft counts of each letter in each word.
# If the soft counts of a letter in the iterated word is not already
# in `letter_soft_counts`, add it (since the soft count values of each letter
# should be consistent across the input text).
for w in wds:
    word_soft_counts = soft_counts(states, w, alphas, betas, alpha_totals)
    for (letter, s1, s2) in word_soft_counts:
        if (letter, s1, s2) not in letter_soft_counts:
            letter_soft_counts[(letter, s1, s2)] = word_soft_counts[(letter, s1, s2)]

pprint(letter_soft_counts)

VerboseFlag = False

# Output of Initialization
if VerboseFlag == True:
    print('-' * 33, '\n')
    print('-\tInitialization\t\t-')
    print('-' * 33)

    for state in states:
        print(f'\nCreating {state.name}')
        print('Transitions')

        for num in range(num_states):
            print(f'\tTo state\t{num}\t{state.a}')

        print('\nEmission probabilities')
        for letter in state.letter_probs:
            print(f'\tLetter\t{letter}\t{round(state.letter_probs[letter], 4)}')

    print('\n', '-' * 33)
    print('Pi:')
    for pi in pi_probs:
        print(f'State\t{pi}\t{round(pi_probs[pi], 4)}')

VerboseFlag = False

# Output of Alphas & Betas
if VerboseFlag == True:
    print('-' * 33, '\n')
    print('-\tIteration Number 0\t-')
    print('-' * 33)

    for w in wds:
        print(f'\n*** word: {w} ***\n')

        print('Alphas:')
        print('Time 1:')
        for state in states:
            print(f'\t{state.name}: {pi_probs[int(state.name[-1])]}')
        for letter in w:
            print(f'Time {w.index(letter) + 2}:')
            for (s, t) in alphas[w]:
                if t == w.index(letter) + 1:
                    print(f'\t{s.name}: {alphas[w][(s, t)]}')

        print('\nBetas:')
        for letter in w:
            print(f'Time {w.index(letter) + 1}:')
            for (s, t) in betas[w]:
                if t == w.index(letter):
                    print(f'\t{s.name}: {betas[w][(s, t)]}')
        print(f'Time {len(w) + 1}:')
        for state in states:
            print(f'\t{state.name}: 1')

        print(f'\nString probability from Alphas: {alpha_totals[w]}')
        print(f'String probability from Betas: {beta_totals[w]}')

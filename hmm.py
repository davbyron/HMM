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
        if hasattr(s, 'pi'):
            alpha[(s, 0)] = s.pi
        else:
            alpha[(s, 0)] = pi[int(s.name[-1])]
    for t in range(1, len(this_word) + 1):
        for to_state in states:
            alpha[(to_state, t)] = 0
            for from_state in states:
                alpha[(to_state, t)] += alpha[(from_state, t - 1)] * from_state.letter_probs[this_word[t - 1]] * from_state.a[int(to_state.name[-1])]
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
                beta[(from_state, t)] += beta[(to_state, t + 1)] * from_state.letter_probs[this_word[t]] * from_state.a[int(to_state.name[-1])]
    return beta

# Calculates the inverse log base 2 of a number `x`.
# Prof. Goldsmith refers to this as the "plog"---hence the function name.
def get_plog(x):
    return -log(x, 2)

# This function calculates the soft counts for each letter of the input word
# for each state transition possible and returns those values in a dictionary.
def soft_counts(states, this_word, alphas, betas, totals):
    sc = {}
    for t in range(0, len(this_word)):
        for from_state in states:
            for to_state in states:
                sc[(this_word[t], from_state, to_state)] = (alphas[this_word][(from_state, t)] * from_state.a[int(to_state.name[-1])] * from_state.letter_probs[this_word[t]] * betas[this_word][(to_state, t + 1)]) / totals[this_word]
    return sc

# Calculates the soft counts for the first letter of each input word
# for each transition possible and returns those values in a dictionary.
def soft_counts_initial(states, this_word, alphas, betas, totals):
    sc = {}
    for t in range(0, 1):
        for from_state in states:
            for to_state in states:
                sc[(this_word[t], from_state, to_state)] = (alphas[this_word][(from_state, t)] * from_state.a[int(to_state.name[-1])] * from_state.letter_probs[this_word[t]] * betas[this_word][(to_state, t + 1)]) / totals[this_word]
    return sc

# (Re-)defines transition probability for each state given the letter soft counts
def calculate_transition(states, softs):
    for from_state in states:
        denominator = 0
        for (letter, s1, s2) in softs:
            if s1 == from_state:
                denominator += softs[(letter, s1, s2)]

        for to_state in states:
            from_state.a[int(to_state.name[-1])] = 0
            numerator = 0
            for (letter, s1, s2) in softs:
                if s1 == from_state and s2 == to_state:
                    numerator += softs[(letter, s1, s2)]
            from_state.a[int(to_state.name[-1])] = numerator / denominator

# (Re-)defines the emission probabilities for each letter for each state
# given the soft counts
def calculate_emission_prob(states, softs):
    for state in states:
        denominator = 0
        for (letter, s1, s2) in softs:
            if s1 == state:
                denominator += softs[(letter, s1, s2)]

        for item in state.letter_probs:
            numerator = 0
            for (letter, s1, s2) in softs:
                if letter == item and s1 == state:
                    numerator += softs[(letter, s1, s2)]
            state.letter_probs[item] = numerator / denominator

# Re-calculates pi for each state given the initial soft counts
def calculate_pi(states, init_softs, words):
    for state in states:
        total = 0
        for (letter, s1, s2) in init_softs:
            if s1 == state:
                total += init_softs[(letter, s1, s2)]
        state.pi = total / len(words)

# Performs a maximization of each state's probability for each letter in the
# corpus. Returns a dictionary with the plog for each word.
def maximization(states, words, alphas, betas, alpha_totals, pis):
    letter_soft_counts = {}
    init_soft_counts = {}

    for w in words:
        word_soft_counts = soft_counts(states, w, alphas, betas, alpha_totals)
        initial_soft_counts = soft_counts_initial(states, w, alphas, betas, alpha_totals)

        for (letter, s1, s2) in word_soft_counts:
            if (letter, s1, s2) not in letter_soft_counts:
                letter_soft_counts[(letter, s1, s2)] = word_soft_counts[(letter, s1, s2)]
            else:
                letter_soft_counts[(letter, s1, s2)] += word_soft_counts[(letter, s1, s2)]
        for (letter, s1, s2) in initial_soft_counts:
            if (letter, s1, s2) not in init_soft_counts:
                init_soft_counts[(letter, s1, s2)] = initial_soft_counts[(letter, s1, s2)]
            else:
                init_soft_counts[(letter, s1, s2)] += initial_soft_counts[(letter, s1, s2)]

    calculate_pi(states, init_soft_counts, words)
    calculate_transition(states, letter_soft_counts)
    calculate_emission_prob(states, letter_soft_counts)

    plogs = {}

    for w in words:
        alphas[w] = forward(states, pi_probs, w)
        betas[w] = backward(states, w)
        alpha_totals[w] = 0

    for w in alphas:
        for (s, t) in alphas[w]:
            if t == len(w):
                alpha_totals[w] += alphas[w][(s, t)]

    for w in alpha_totals:
        plog = get_plog(alpha_totals[w])
        plogs[w] = plog

    return plogs

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
    state.a = {}
    for num in range(num_states):
        state.a[num] = round(1 / num_states, 4)
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
    for (s, t) in alphas[w]:
        if t == len(w):
            alpha_totals[w] += alphas[w][(s, t)]

# Calculates string probability given its betas.
for w in betas:
    for (s, t) in betas[w]:
        if t == 0:
            beta_totals[w] += pi_probs[int(s.name[-1])] * betas[w][(s, t)]

# Calculates the plog of each word.
'''for w in alpha_totals:
    plog = get_plog(alpha_totals[w])
    plogs[w] = plog'''

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

VerboseFlag = False

# Output of plogs
if VerboseFlag == True:
    for word in plogs:
        print(f'{word[:-1]}\t\t{round(plogs[word], 4)}')
    print(f'\nSum of all plogs: {round(sum(plogs.values()), 4)}')

VerboseFlag = False

# Output of soft counts
if VerboseFlag == True:
    for c in sorted(letters):
        for (letter, s1, s2) in letter_soft_counts:
            if letter == c:
                print(f'{letter}\t{s1.name[-1]}\t{s2.name[-1]}\t{round(letter_soft_counts[(letter, s1, s2)], 4)}')

VerboseFlag = True

# Output of Maximization
if VerboseFlag == True:
    all_plog_values = []
    previous_plog = 0

    for i in range(100):
        plg = maximization(states, wds, alphas, betas, alpha_totals, pi_probs)

        if i == 0:
            previous_plog = sum(plg.values())
        else:
            new_plog = sum(plg.values())
            if new_plog > previous_plog or previous_plog - new_plog < 0.1:
                break
            previous_plog = new_plog

        print('-' * 33, '\n')
        print(f'-\tIteration {i + 1}\t\t-')
        print('-' * 33)

        print('Pi:')
        for state in states:
            print(f'\t{state.name}: {round(state.pi, 4)}')

        print('\n*** Emission Probabilities ***')
        for state in states:
            print(state.name)
            state_letter_probs_sorted = sorted(state.letter_probs.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
            for (l, v) in state_letter_probs_sorted:
                print(f'\t{l}\t{round(v, 4)}')

        letter_logs = {}
        for letter in letters:
            letter_log_calc = []
            total = 0
            for state in states:
                letter_log_calc.append(state.letter_probs[letter])
            for state_letter_prob in letter_log_calc:
                if total == 0:
                    total += state_letter_prob
                else:
                    total = total / state_letter_prob
            letter_logs[letter] = log(total, 2)
        positives = [(l, v) for (l, v) in sorted(letter_logs.items(), key = lambda kv:(kv[1], kv[0]), reverse=True) if v > 0]
        negatives = [(l, v) for (l, v) in sorted(letter_logs.items(), key = lambda kv:(kv[1], kv[0])) if v < 0]

        print(f'\nLog ratios of emissions from the {len(states)} states:')
        for (l, v) in positives:
            print(f'{l}\t{round(v, 4)}')
        print('')
        for (l, v) in negatives:
            print(f'{l}\t{round(v, 4)}')

        print('\n*** Transition Probabilities ***')
        for from_state in states:
            print(f'From {from_state.name}:')
            for to_state in states:
                print(f'\tTo {to_state.name}: {from_state.a[int(to_state.name[-1])]}')

        all_plog_values.append(sum(plg.values()))
        print(f'\nSum of plogs: {sum(plg.values())}')

    print(all_plog_values.index(min(all_plog_values)) + 1)

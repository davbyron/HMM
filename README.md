# Hidden Markov Model

> First steps to building my own Hidden Markov Model for predicting the probability of sequences of characters/words/strings in the English language.

This script was produced using Python 3.7.2 and was originally submitted as homework for LING 38600 'Computational Linguistics' taught by John Goldsmith at the University of Chicago, spring quarter 2019.

This repository contains the following files:

- `hmm_1.py`: This script constitutes the first steps toward building my HMM. It calculates the probability assigned to a string by a probabilistic Finite-State Automata. All details are provided within the comments of the code.
- `hmm_2.py`: This script builds on `hmm_1.py` by calculating the soft counts for each word in the input file. That is, it calculates the probability of emitting each letter in each word for each possible state transition.
- `sample_text_hmm_1.txt`: Text file containing the nonse words *babi#* and *dida#*. This file can (and was) used to test the code on a small-scale level.
- `english_1000.txt`: Text file containing 1000 English tokens for a larger-scale test of the code. This file was provided by Professor Goldsmith.

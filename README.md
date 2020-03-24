# Hidden Markov Model

> This script was produced using Python 3.7.2 and was originally submitted as an assignment for LING 38600 'Computational Linguistics' taught by John Goldsmith at the University of Chicago, spring quarter 2019.

This Hidden Markov Model (HMM) probabilistically determines consonants from vowels in any Latin alphabet-using language given a set of words from that language.

## Usage

> NOTE: This HMM uses data from the file `english_1000.txt`. If you would like to use your own data, the format of the file must (a) be a `.txt` file, (b) have one word per line, and (c) have each line end in `#`. You can then manually change `english_1000.txt` in `hmm.py` to your own file.

In the command line, type `python hmm.py`. The command line will then prompt you to enter the number of "states" you would like to use. Any number of states is acceptable, but each additional state greatly increases  processing time.

#### Sample Output

In the sample output below, the algorithm has done an alright job at differentiating vowels and consonants, grouping the former in "Set 2" and the latter in "Set 1".

```
$ python hmm.py
How many states would you like to use?: 2

I tried distinguishing consonants and vowels...

Set 1: b c d f g j k m n p q r s t v w y z
Set 2: # ' . a e h i l o u x

... How'd I do?
```

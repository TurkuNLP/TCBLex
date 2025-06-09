# TCBLex

## Main repository for Turku Children's Book Lexicon

This repository contains scripts that were used in building TCBLex from a corpus of CoNLLU -files as well as a Python Notebook that can be used to view statistical data from TCBLex.

To see the pipeline that was used to build the corpus TCBLex is based on, please view the repository for the [Turku Children's Book Corpus pipeline](https://github.com/TurkuNLP/TCBC-pipeline)

## Interpreting TCBLex

TCBLex is released as a collection of CSV-files, where the character ";" is used as a separator.

Note that '15' in the context of TCBLex always means "intended reading ages from 15 to 18". We use '15' for ease of coding.

The lexical database is split into four different folders:
  - ages_csv, which contain sublexicons for each intended reading age separately
  - groups_csv, which contains sublexicons for each age group
  - registers_csv, which contains sublexicons for each register
  - whole_csv, which contains the whole lexicon

Each CSV-file has the same exact columns in the same order. For more details on how each of the statistics is calculated, please refer to the paper.
  - text (Word type)
  - lemma (Lemma of the word type)
  - upos (POS-tag)
  - Word-POS F (Frequency of the word type and POS-tag bigram)
  - Word F (Frequency of word type)
  - Word CD (Contextual diversity of word type)
  - Word D (Dispersion of word type)
  - Word U (Estimated frequency per 1 million words for word type)
  - Word SFI (Standardized Frequency Index of word type)
  - Word Zipf (Zipf value of word type)
  - Lemma F (Frequency of lemma)
  - Lemma CD (Contextual diversity of lemma)
  - Lemma D (dispersion of lemma)
  - Lemma U (Estimated frequency per 1 million words for lemma)
  - Lemma SFI (Standardized Frequency Index of lemma)
  - Lemma Zipf (Zipf value of lemma)
  - Lemma IFS (Inflection family size of lemma)
  - Word Syllables (Number of syllables in word type)
  - Word Length (Length of word type in characters)
  - Lemma Length (Length of lemma in characters)
  - Word FSA (Intended reading age word type is first encountered at)
  - Lemma FSA (Intended reading age lemma is first encountered at)

### Alternative version of TCBLex
For the versions with features, the "upos" and "Word-POS F" are replaced with "upos+features" and "Word-POS+FEATS F" respectively.

These versions include extra information on word types, such as what is the case or tense of the word. These features also follow the Universal Dependencies framework. For more information, see [the webpage for Finnish UD](https://universaldependencies.org/fi/index.html).

## Viewing statistical data of TCBLex

To use Python Notebook provided in this repository, you must first download TCBLex and place the "Data" folder in the same folder as the notbook. After this, you can utilize the functions and examples provided in the notebook to display the same plots seen in the article or get the same statistical data as presented in the tables of the article. The code in the notebook has been commented to make viewing and using it easier.

# About

A simple dictionary-based tool for obtaining the sentiment score of a sentence based on SentiWordNet 3.0. Sentiment scores are between -1 and 1, greater than 0 for positive and less than 0 for negative.

Dictionary-based sentiment analysis does not perform as well as a trained classifier, but it is domain-independent, based on *a priori* knowledge of words' sentiment values.

Negations and multiword expressions are handled.

# Dependencies

NLTK

# Usage

First download SentiWordNet 3.0 [here](http://sentiwordnet.isti.cnr.it/), and delete any header lines so that the file contains only data, e.g.

a	00001740	0.125	0	able#1	(usually followed by `to') having the necessary...

The SentimentAnalysis constructor must be passed the name of the SentiWordNet text file and your choice of weighting across word senses.

```python
s = SentimentAnalysis(filename='SentiWordNet.txt',weighting='geometric')
>>> s.score('I love you!')
0.59375
>>> s.score('Pants are the worst.')
-0.125
>>> s.score('I do not particularly enjoy this product.')
-0.15885416666666666
```

## Options

Barring any word sense disambiguation, you can use do an arithmetic, geometric or harmonic mean across the senses by passing in 'average','geometric' or 'harmonic' to the constructor. This choice can have an important effect on accuracy, see Guerini et al. "Sentiment Analysis: How to Derive Prior Polarities from SentiWordNet".


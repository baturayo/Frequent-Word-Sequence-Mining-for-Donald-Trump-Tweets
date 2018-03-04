# Frequent Word-Sequence Mining for Donald Trump Tweets

The aim of this project is to find most frequent word sequences in Donald
Trump Tweets. It outputs most frequent n word patterns. E.G "Make America Great Again" substring is the most used
substring in his tweets for 4-word sequences. You can also see the patterns for 1,2,3 and 4 word sequences in this project.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

To run this Python 2.7 project, you need to install the python libraries below to run the project

```
pip install wordcloud
python -mpip install -U matplotlib
```

### Data
Each line symbolises one transaction and the words stand for the items.

### Running the Code

* Change Directory into Frequent Word Sequence Mining folder

* Paste the following script to run the code
```
python sequential_words_analyser_apriori.py -f DATASET -s minSupport -n topNMostFreqSequences
```
* DATASET: Input data where there are transactions
* minSupport: optional parameter and default value is
 0.02. It is the minimum support value for Apriori algorithm <br />
* topNmostFreqSequences: optional parameter and default value is 5. The parameter will
print the top N most frequent word sequences.
<br />

Example script for TrumpTweets data:
```
python sequential_words_analyser_apriori.py -f TrumpTweets.txt
 -s 0.02 -n 6
```
## Algorithm Steps
Generalized Sequential Pattern algorithm is used to analyze
frequent word-sequences. The algorithm is very
similar with Apriori algorithm.

1. Read transactions from input file line by line
2. Create variable n and set n = 1 which stands for the n word sequences
 (if n=2 then the candidate items should have 2 words e.g
'frequent words')
3. Find unique 1-word-items from transactions and create an item set array
4. Divide item set into chunks and multiprocess each chunk to find frequency
5. To find frequency add ' ' char to the beginning and end of both item and transaction. It will provide
us to check whether the item is a subset of transaction. (e.g item = ' make america '
transaction = ' make america great again '.) If we do not add white spaces, then the algorithm may over count
some different patterns such as item = 'a book' transaction = 'America book' then there will be a false match.
Instead of n-gram, I prefer this technique to save memory space.
6. Filter items from item set that their frequency value is less than the minimum support (multi processing)
7. Generate candidate n+1 word sequences set by combining filtered n word sequences and filtered 1-word-item set
8. Find frequency of candidate set (same multi processing technique as step 4)
9. Filter candidate set based on minimum support value (multi process)
10. n += 1
11. If filtered candidate set has an item go back to step 7 else plot and print the results

## Output

* The algorithm prints the top N most frequent word-sequences.
* It also visualise the most frequent word-sequences in 'most_freq_words_n.jpg' files.
N symbolise the number of words in sequence. E.G if you open most_freq_words_4.jpg file
you will see that the most frequent sequence is 'make america great again'.


## Author

* **Baturay Ofluoglu**

## Acknowledgments

* Advance Database Research and Modelling (ADREM) group to provide data

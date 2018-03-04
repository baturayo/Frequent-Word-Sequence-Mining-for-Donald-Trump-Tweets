"""
Description     : Python implementation of the Apriori Algorithm to Analyze frequent sequential words

Usage:
    $python apriori.py -f DATASET -s minSupport -n topNMostFreqSequences

    $python apriori.py -f DATASET -s 0.15 -n 6
"""

import sys
from collections import defaultdict, OrderedDict
import itertools
from multiprocessing import Pool, cpu_count
import multiprocessing
from functools import partial
import math
from optparse import OptionParser
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from time import time

def Counter(itemSet, transactionList):
    """
        If an item is a subset of a transaction add 1 to localset[item] to keep trace of freq for each item
        Return:
         - localset: frequecy dictionary of each item that has N words
    """
    localSet = defaultdict(int)
    for item in itemSet:
        ''' Add white space to the beginning and end of both transactions and items
            E.g item = ' a ', transaction = ' apriori ' then item will not be a subset of transaction because 
            we add a white space
        '''
        item = ' ' + item + ' '
        for transaction in transactionList:
            transaction = ' ' + transaction + ' '
            if item in transaction:
                # Add 1 to the item if item is a subset of transaction and then remove the white space added to the end
                localSet[item[1:-1]] += 1
    return localSet


def MinSupportFilter(localSet, numberOfTransactions):
    """
       run MinSupportFilter to filter the items that cannot satisfy minimum support threshold
       Input:
        - localSet: localSet is a dictionary includes the N words items
                    e.g for 1 word items= {'liar' : 20, 'execut': 19}
        - numberOfTransactions: Integer value that has the total number of transactions in file
       Return:
        - _itemSet: dictionary object and filtered version of localset based on minimum support threshold
    """
    _itemSet = list()
    for item, count in localSet.items():
        support = float(count) / numberOfTransactions
        if support >= minSupport:
            _itemSet.append(item)
    return _itemSet


def chunks(l, n):
    """
    A generator function for chopping up a given list l into chunks of
    length n. It will be used by multiprocessing tool to divide the list objects
    """
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def returnItemsWithMinSupport(itemSet, transactionList, numberOfTransactions):
    """
       Calculates the support for items in the itemSet by using multiprocessing tool and returns a subset
       of the itemSet each of whose elements satisfies the minimum support

       Input:
        - itemSet: list that includes all unique word sequences in the input file. Number of words in itemSet may differ
                    e.g = ['liar', 'execut', 'forget']
                    e.g = ['liar execut', 'forget me']
        - transactionList: list that has all the transactions and each lines symbolise one transaction in the input file.
                           e.g = ['my name is Baturay', 'I love Data Science']
       Return:
        - _freqSet: dictionary object includes all the frequency values of itemSet input list.
        - itemSetWithMinSupport: list object that have filtered items from itemSet input list based on minimum support value
    """

    # Build a pool of N processes
    pool = multiprocessing.Pool(processes=cpu_count(), )

    # Fragment the string data into N chunks
    if len(itemSet) / cpu_count() > 0:
        N = math.ceil(len(itemSet) / float(cpu_count()))
        N = int(N)
    else:
        N = 1

    # Divide itemSet into chunks to process with different cores
    partitioned_itemSet = list(chunks(itemSet, N))

    # Create multiple inputs for Counter
    multiCount = partial(Counter, transactionList=transactionList)

    # Generate localSet and freqSet by using multiple processing
    freqSet = pool.map(multiCount, partitioned_itemSet)
    pool.close()

    # Build again a pool of N processes
    pool = multiprocessing.Pool(processes=cpu_count(), )

    # Multiple inputs for MinSupportFilter to prune the itemSet based on minimum support
    minSupportFilter = partial(MinSupportFilter, numberOfTransactions=numberOfTransactions)

    # Eliminate itemSet based on minSupport by using multiple processing
    _itemSet = pool.map(minSupportFilter, freqSet)
    pool.close()

    # Merge multiple core results of _itemset into one dimensional list
    itemSetWithMinSupport = list()
    for chunk in _itemSet:
        itemSetWithMinSupport = itemSetWithMinSupport + chunk
    del _itemSet

    # Merge multiple core results of _freqset into one dimensional list
    _freqSet = dict()
    for chunk in freqSet:
        _freqSet.update(chunk)

    return _freqSet, itemSetWithMinSupport


def joinSet(itemSet, currentLSet):
    """
       The method takes cartesian product of itemSet and currentLSet.
       Input:
        - itemSet: list that includes all unique words in the input file (only one word per item).
                    e.g = ['liar', 'execut', 'forget']
        - currentLSet: list that has all the filtered current state items with minimum support. The number of words per
                       item in this list may change between 1 to N depending on minimum support value
                       e.g (2 words per item) = ['make america', 'great again']
       Return:
        - the combination of itemSet and currentLSet
    """
    return map(' '.join, itertools.chain(itertools.product(currentLSet, itemSet)))


def getItemSetTransactionList(data_iterator):
    """
       It iterates over the input dataset and generate transactions and itemSet
       Input:
       - data_iterator: generator to read each line of the input file

       Return:
       - itemSet: list that includes all unique words in the input file (only one word per item).
                  e.g = ['liar', 'execut', 'forget']
       - transactionList: list object that has tweet strings as an element
    """
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        record = record.replace('\n', '')  # Remove \n chars if exists
        for item in record.split(' '):
            itemSet.add(item)  # Generate 1-itemSets
        transactionList.append(str(record))

    try:
        itemSet.remove('')  # Remove '' transactions if exists
    except:
        pass
    return list(itemSet), transactionList


def runApriori(data_iter):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items
     - frequency sets of items
     - max number of word combinations
    """
    itemSet, transactionList = getItemSetTransactionList(data_iter)
    freqSet = defaultdict(int)
    largeSet = dict() # Dictionary which stores (key=n-itemSets,value=support)
    numberOfTransactions = len(transactionList) # Calculate number of tweets or transactions

    print('There are %s transactions.' % numberOfTransactions)
    localFreqSet, oneCSet = returnItemsWithMinSupport(itemSet, transactionList, numberOfTransactions)

    currentLSet = oneCSet
    k = 2 # k-itemSets or k-word sequences
    while currentLSet:
        freqSet[k - 1] = localFreqSet
        largeSet[k - 1] = currentLSet
        currentLSet = joinSet(oneCSet, currentLSet)
        localFreqSet, currentCSet = returnItemsWithMinSupport(currentLSet,
                                                              transactionList, numberOfTransactions)
        currentLSet = currentCSet
        print('%s word combinations is in progress..' % k)
        k += 1

    return largeSet, freqSet, k - 1


def plotWordFrequencies(freqs, N):
    """
        It visualises the most frequent word sequences that has 1 to N words
        See most_freq_words_1.jpg or
            most_freq_words_n.jpg where n is an integer
    """
    for i in range(1, N):
        wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(freqs[i])
        plt.figure(figsize=(20, 10), facecolor='k')
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig('most_freq_words_%s.jpg' % i)


def dataFromFile(fname):
    """Function which reads from the file and yields a generator"""
    file_iter = open(fname, 'rU')
    for line in file_iter:
        yield line

def printResults(freqs, K, N):
    """
        Print top N most frequent K-word sequences
    """

    for k in range(1, K):
        print('#####################################')
        print('The most frequent %s-word substrings:' % k)
        print('#####################################')
        sortedFreq = sorted(((value, key) for (key, value) in freqs[k].items()), reverse=True)
        for i in range(N):
            print ('%s : %s' % (sortedFreq[i][1], sortedFreq[i][0]))


if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename containing txt',
                         default=None)
    optparser.add_option('-s', '--minSupport',
                         dest='minS',
                         help='minimum support value',
                         default=0.02,
                         type='float')
    optparser.add_option('-n', '--topNMostFreqSequences',
                         dest='topN',
                         help='top n most frequent substring sequences',
                         default= 5,
                         type='int')

    (options, args) = optparser.parse_args()

    inFile = None
    if options.input is None:
            inFile = sys.stdin
    elif options.input is not None:
            inFile = dataFromFile(options.input)
    else:
            print 'No dataset filename specified, system with exit\n'
            sys.exit('System will exit')

    minSupport = options.minS
    start_time = time()
    items, freqs, k = runApriori(inFile)
    elapsed_time = time() - start_time
    print('Apriori running time: %s' % elapsed_time)
    start_time = time()
    plotWordFrequencies(freqs, k)
    elapsed_time = time() - start_time
    print('Most Frequent Word Visualisation running time: %s' % elapsed_time)
    printResults(freqs, k, options.topN)


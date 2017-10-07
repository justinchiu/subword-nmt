#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Not Rico Sennrich

from __future__ import unicode_literals

import sys
import codecs
import re
import copy
import argparse
from collections import defaultdict, Counter, namedtuple

# hack for python2/3 compatibility
from io import open
argparse.open = open

Vocab = namedtuple(
    'Vocab',
    [
        'sens',
        'unigram_counts',
        'unigram_to_senidxs',
        'bigram_counts',
        'bigram_to_senidxs',
        'unigram_to_bigrams',
    ]
)

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input text (default: standard input).")
    parser.add_argument(
        '--codes', '-c', type=argparse.FileType('w'), default=sys.stderr,
        metavar='PATH',
        help="Output file for BPE codes (default: standard error)")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=None,
        metavar='PATH',
        help="Output file for transformed corpus (default: None)")
    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a n-gram) (default: %(default)s))")
    parser.add_argument(
        '--min-frequency', type=int, default=2, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s))')
    parser.add_argument('--dict-input', action="store_true",
        help="If set, input file is interpreted as a dictionary where each line contains a word-count pair")
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="verbose mode.")

    return parser

def get_vocabulary(fobj, is_dict=False):
    """Read text and return dictionary that encodes vocabulary
    """
    sens = []
    unigram_counts     = Counter()
    unigram_to_senidxs = defaultdict(list)
    bigram_counts      = Counter()
    bigram_to_senidxs  = defaultdict(list)
    unigram_to_bigrams = defaultdict(set)
    for i, line in enumerate(fobj):
        sen = line.strip().split()
        sens.append(sen)
        #for word in sen:
        #    unigram_counts[word] += 1
        #    unigram_to_senidxs[word].append(i)
        for bigram in zip(sen, sen[1:]):
            bigram_counts[bigram] += 1
            bigram_to_senidxs[bigram].append(i)
            #unigram_to_bigrams[bigram[0]].add(bigram)
            #unigram_to_bigrams[bigram[1]].add(bigram)
    return Vocab(
        sens=sens,
        unigram_counts=unigram_counts,
        unigram_to_senidxs=unigram_to_senidxs,
        bigram_counts=bigram_counts,
        bigram_to_senidxs=bigram_to_senidxs,
        unigram_to_bigrams=unigram_to_bigrams
    )

def update_pair_statistics(pair, changed, stats, pointers):
    pass

def replace_pair(pair, vocab):
    unigram_counts     = vocab.unigram_counts
    unigram_to_senidxs = vocab.unigram_to_senidxs
    bigram_counts      = vocab.bigram_counts
    bigram_to_senidxs  = vocab.bigram_to_senidxs
    first, second = pair
    pair_str = '++'.join(pair).replace('\\', '\\\\')
    changes = []
    senidxs = vocab.bigram_to_senidxs[pair]
    sens = vocab.sens
    for senidx in senidxs:
        sen = sens[senidx]
        senlen = len(sen)
        occurrences = []
        i = 0
        while i < senlen - 1:
            if sen[i] == first and sen[i+1] == second:
                # Found a bigram
                if i > 1 and sen[i-2] == first and sen[i-1] == second:
                    # Something like AB AB
                    # Previous word will be compressed, so count it!
                    bigram_counts[(pair_str, pair_str)] += 1
                    # but also remove B A
                    bigram_counts[(second, first)] -= 1
                    if bigram_counts[(second, first)] < 0:
                        del bigram_counts[(second, first)]
                    left_bigram = (second, first)
                elif i > 0:
                    left_word = sen[i-1]
                    # add new left bigram
                    new_left_bigram = (left_word, pair_str)
                    bigram_counts[new_left_bigram] += 1
                    bigram_to_senidxs[new_left_bigram].append(senidx)
                    # remove old left bigram
                    left_bigram = (left_word, first)
                    if left_bigram in bigram_counts:
                        bigram_counts[left_bigram] -= 1
                else:
                    # No bigram on the left
                    left_bigram = None
                if i+3 < senlen and sen[i+2] == first and sen[i+3] == second:
                    # Need to be careful of not double counting though
                    # right now this is undercounting, i.e. if we have AB AB
                    # this will not count AB,AB since it has not been processed yet
                    # We want to skip this case because it will be processed
                    # in the next iteration of the loop.
                    right_bigram = None
                elif i+2 < senlen:
                    right_word = sen[i+2]
                    # add new right bigram
                    new_right_bigram = (pair_str, right_word)
                    bigram_counts[new_right_bigram] += 1
                    bigram_to_senidxs[new_right_bigram].append(senidx)
                    # remove old right bigram
                    right_bigram = (second, right_word)
                    if right_bigram in bigram_counts:
                        bigram_counts[right_bigram] -= 1
                else:
                    right_bigram = None
                occurrences.append((i, left_bigram, right_bigram))
                i += 2
            else:
                i += 1
        for (i, left_bigram, right_bigram) in occurrences[::-1]:
            # mutate sentence by removing bigram and adding new unigram
            del sen[i:i+2]
            sen.insert(i, pair_str)
            # remove bigram to senidxs pointers
            if left_bigram in bigram_to_senidxs:
                idx_to_delete = bigram_to_senidxs[left_bigram].index(senidx)
                del bigram_to_senidxs[left_bigram][idx_to_delete]
            if right_bigram in bigram_to_senidxs:
                idx_to_delete = bigram_to_senidxs[right_bigram].index(senidx)
                del bigram_to_senidxs[right_bigram][idx_to_delete]

        # remove bigram statistics
        del vocab.bigram_counts[pair]

def main(infile, codefile, outfile, num_symbols, min_frequency=2, verbose=False, is_dict=False):
    """Learn num_symbols BPE operations from vocabulary, and write to outfile.
    """

    # version 0.2 changes the handling of the end-of-word token ('</w>');
    # version numbering allows bckward compatibility
    codefile.write('#version: 0.3\n')

    vocab = get_vocabulary(infile, is_dict)
    stats = vocab.bigram_counts
    for i in range(num_symbols):
        # this is super sad, will write a mutable heap implementation
        most_frequent = vocab.bigram_counts.most_common(1)[0][0]
        if stats[most_frequent] < min_frequency:
            sys.stderr.write('#no pair has frequency >= {0}. Stopping\n'.format(min_frequency))
            break

        if verbose:
            sys.stderr.write('#pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'
                .format(i, most_frequent[0], most_frequent[1], stats[most_frequent]))
        codefile.write('{0} {1}\n'.format(*most_frequent))
        changes = replace_pair(most_frequent, vocab)
        #update_pair_statistics(most_frequent, changes, stats, indices)
        #del stats[most_frequent]
        if False and not i % 100:
            # seriously, what the fuck is this?
            prune_stats(stats, big_stats, threshold)
    for sen in vocab.sens:
        outfile.write(' '.join(sen))
        outfile.write('\n')


if __name__ == '__main__':
    # python 2/3 compatibility
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    else:
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)
        pass

    parser = create_parser()
    args = parser.parse_args()

    # read/write files as UTF-8
    if args.input.name != '<stdin>':
        args.input = codecs.open(args.input.name, encoding='utf-8')
    if args.codes.name != '<stderr>':
        args.codes = codecs.open(args.codes.name, 'w', encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')
    main(args.input, args.codes, args.output, args.symbols, args.min_frequency, args.verbose, is_dict=args.dict_input)
    #import profile; profile.run('main(args.input, args.output, args.symbols, args.min_frequency, args.verbose, is_dict=args.dict_input)')

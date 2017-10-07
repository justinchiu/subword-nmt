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
from heappy import PQ

# hack for python2/3 compatibility
from io import open
argparse.open = open

Vocab = namedtuple(
    'Vocab',
    [
        'sens',
        'id',
        'unigram_to_id',
        'id_to_unigram',
        'bigram_counts',
        'bigram_to_senidxs',
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
    unigram_to_id     = {}
    id_to_unigram     = []
    id                = 0
    bigram_counts     = PQ()
    bigram_to_senidxs = defaultdict(list)
    for i, line in enumerate(fobj):
        sen = line.strip().split()
        sens.append(sen)
        for word in sen:
            if not word in unigram_to_id:
                unigram_to_id[word] = id
                id_to_unigram.append(word)
                id += 1
        for b in zip(sen, sen[1:]):
            first = unigram_to_id[b[0]]
            second = unigram_to_id[b[1]]
            bigram_counts.increment(first, second)
            bigram_to_senidxs[b].append(i)
    return Vocab(
        sens=sens,
        id=[id],
        unigram_to_id=unigram_to_id,
        id_to_unigram=id_to_unigram,
        bigram_counts=bigram_counts,
        bigram_to_senidxs=bigram_to_senidxs,
    )

def update_pair_statistics(pair, changed, stats, pointers):
    pass

def replace_pair(pair, vocab):
    bigram_counts      = vocab.bigram_counts
    bigram_to_senidxs  = vocab.bigram_to_senidxs
    unigram_to_id= vocab.unigram_to_id
    first, second = pair
    pair_str = '++'.join(pair).replace('\\', '\\\\')
    unigram_to_id[pair_str] = vocab.id[0]
    vocab.id_to_unigram.append(pair_str)
    vocab.id[0] += 1
    changes = []
    senidxs = vocab.bigram_to_senidxs[pair]
    sens = vocab.sens
    pair_id = vocab.unigram_to_id[pair_str]
    first_id = vocab.unigram_to_id[first]
    second_id = vocab.unigram_to_id[second]
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
                    bigram_counts.increment(pair_id, pair_id)
                    # but also remove B A
                    bigram_counts.decrement(second_id, first_id)
                    left_bigram = (second, first)
                elif i > 0:
                    left_word = sen[i-1]
                    left_word_id = unigram_to_id[left_word]
                    # add new left bigram
                    new_left_bigram = (left_word, pair_str)
                    bigram_counts.increment(left_word_id, pair_id)
                    bigram_to_senidxs[new_left_bigram].append(senidx)
                    # remove old left bigram
                    left_bigram = (left_word, first)
                    bigram_counts.decrement(left_word_id, first_id)
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
                    right_word_id = unigram_to_id[right_word]
                    # add new right bigram
                    new_right_bigram = (pair_str, right_word)
                    bigram_counts.increment(pair_id, right_word_id)
                    bigram_to_senidxs[new_right_bigram].append(senidx)
                    # remove old right bigram
                    right_bigram = (second, right_word)
                    bigram_counts.decrement(second_id, right_word_id)
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
        most_frequent = vocab.bigram_counts.top()
        stats.pop()
        most_frequent = (vocab.id_to_unigram[most_frequent[0]], vocab.id_to_unigram[most_frequent[1]])
        if stats.topCount() < min_frequency:
            sys.stderr.write('#no pair has frequency >= {0}. Stopping\n'.format(min_frequency))
            break
        if verbose:
            sys.stderr.write('#pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'
                .format(i, most_frequent[0], most_frequent[1], stats[most_frequent]))
        codefile.write('{0} {1}\n'.format(*most_frequent))
        changes = replace_pair(most_frequent, vocab)
        #update_pair_statistics(most_frequent, changes, stats, indices)
    if outfile:
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
    if args.output and args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')
    main(args.input, args.codes, args.output, args.symbols, args.min_frequency, args.verbose, is_dict=args.dict_input)
    #import profile; profile.run('main(args.input, args.codes, args.output, args.symbols, args.min_frequency, args.verbose, is_dict=args.dict_input)')

# Copyright 2017 Rice University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import tensorflow as tf
import numpy as np
import random
import os
import logging as log
import time
import pickle
from filter_unknown import filter_unknown_vocabs
from utils import sample
from datetime import datetime
from data_reader import JsonParser
from model import Model
from lda import LDA

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('data_file', type=str, nargs=1,
                       help='input file in JSON')
    argparser.add_argument('--save_dir', type=str, default='save',
                       help='directory where model is stored')
    argparser.add_argument('--debug', action='store_true',
                       help='enable printing debug log to debug.log')
    argparser.add_argument('--seed', type=int, default=None,
                       help='random seed')
    argparser.add_argument('--num_samples_seqs', type=int, default=10,
                       help='number of samples of sequences used in estimator')
    argparser.add_argument('--num_samples_topics', type=int, default=10,
                       help='number of samples of topics used in estimator')
    argparser.add_argument('--num_iters', type=int, default=10,
                       help='number of iterations for convergence in estimators')
    argparser.add_argument('--location_sensitive', action='store_true',
                       help='document is location-sensitive set of calls (default False)')
    args = argparser.parse_args()
    if args.seed is None:
        args.seed = int(time.time())
    random.seed(args.seed)

    if args.debug:
        log.basicConfig(level=log.DEBUG, filename='debug.log', filemode='w', format='%(message)s')
    start = datetime.now()
    print('Started at {}'.format(start))

    with tf.Session() as sess:
        with open(args.data_file[0]) as data:
            parser = JsonParser(data)

        kld = KLD(args, parser, sess)
        filter_unknown_vocabs(parser.json_data, set(kld.vocab.keys()))
        for pack in parser.packages:
            print('### ' + pack['name'])
            log.debug('\n### ' + pack['name'])
            klds = [(l, kld.compute(l, pack)) for l in set(parser.locations(pack))]
            for l, k in sorted(klds, key=lambda x: -x[1]):
                print('  {:35s} : {:.2f}'.format(l, k))

    if len(kld.not_in_vocab) > 0: print('Not in vocab:', ", ".join(map(repr, kld.not_in_vocab)))
    print('Time taken: {}'.format(datetime.now() - start))

class KLD():
    def __init__(self, args, parser, sess):
        self.args = args
        self.parser = parser
        self.sess = sess
        self.not_in_vocab = []

        with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
            saved_args = pickle.load(f)
        with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
            self.chars, self.vocab = pickle.load(f)

        self.model = Model(saved_args, True)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables()) #tf.all_variables
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)

        with open(os.path.join(args.save_dir, 'lda.pkl'), 'rb') as f:
            self.lda = LDA(from_file=f)

    def inner_sum(self, seq, seqs, topics):
        P = float(seqs.count(seq)) / len(seqs)
        Q = sum(self.reference(seq, topic) for topic in topics) / len(topics)
        return (np.log(P) - np.log(Q)
                - self.sequence_bias(seq, seqs)
                - self.topic_bias(topics)) if P > 0 and Q > 0 else 0

    def reference(self, seq, topic):
        """ Compute the reference probability of a sequence given a topic """
        stream = self.parser.stream(seq)
        pr = self.model.probability(self.sess, stream, topic, self.vocab)
        pr = pr[:-1] # distribution is over the next symbol, so ignore last
        stream = stream[1:] # first symbol is prime (START), so ignore it
        probs = [p[self.vocab[char]] for p, char in zip(pr, stream)]
        log.debug(topic)
        log.debug(stream)
        log.debug(probs)
        return np.prod(probs)

    def topic_bias(self, topics):
        """ TODO """
        return 0.

    def sequence_bias(self, seq, seqs):
        """ bias is half of the negative variance of the estimate. the variance
            is itself estimated through bootstrap resampling."""
        values = []
        for i in range(self.args.num_iters):
            values.append(float(seqs.count(seq)) / len(seqs))
            seqs = sample(seqs, nsamples=len(seqs))
        avg = float(sum(values)) / self.args.num_iters
        var = [(value - avg) ** 2 for value in values]
        var = sum(var) / (self.args.num_iters - 1)
        return -var / 2.

    def compute(self, l, pack):
        seqs_l = list(self.parser.sequences(pack, l))
        if len(seqs_l) == 0:
            return -3. # empty list
        samples = [sample(seqs_l, nsamples=1) for i in range(self.args.num_iters)]
        bow = set((event['call'], event['location'] if self.args.location_sensitive else None)
                for seq in samples for event in seq)
        lda_data = [';'.join([call for (call, _) in bow if not call == 'TERMINAL'])]
        not_in_vocab = [call for (call, _) in bow if call not in self.vocab]
        if not not_in_vocab == []:
            self.not_in_vocab += list(set(not_in_vocab) - set(self.not_in_vocab))
            return -1. # elems not in vocab

        log.debug('\n' + l)
        triple_sample = [(samples[i], # for each sample, sample num_samples_seqs samples from seqs_l
                          sample(seqs_l, nsamples=self.args.num_samples_seqs),
                          self.lda.infer(lda_data, nsamples=self.args.num_samples_topics)[0])
                                for i in range(self.args.num_iters)]
        K = list(map(lambda t: self.inner_sum(*t), triple_sample))
        c = np.count_nonzero(K) # discard inner_sums that resulted in 0
        return sum(K) / c if c > 0 else -2.

if __name__ == '__main__':
    main()

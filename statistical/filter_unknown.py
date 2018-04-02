# Copyright 2018 Rice University
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
import numpy as np
import random
import os
import pickle
import json

def parse_vocabs(filename):
    with open(filename, 'rb') as f:
        vocabs = pickle.load(f)[1]
    return set(vocabs.keys())



def filter_unknown_vocabs(json_data, vocabs):
    def check_seq(seq):
        events = seq['sequence']
        events = list(filter(lambda x: x['call'] in vocabs, events))
        seq['sequence'] = events
        return len(events) > 0 and (len(events) > 1 or events[0]['call'] != "TERMINAL")

    for pkg in json_data['packages']:
        pkg['data'] = list(filter(check_seq, pkg['data']))

def do_filter(infile, outfile, dirname):
    vocabs = parse_vocabs(os.path.join(dirname, 'chars_vocab.pkl'))
    print(vocabs)
    with open(infile) as f:
        data = json.load(f)
    filter_unknown_vocabs(data, vocabs)
    with open(outfile, 'w') as f:
        json.dump(data, f)

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('in_file', help='input JSON file')
    argparser.add_argument('out_file', help='output JSON file')
    argparser.add_argument('-d', dest="save_dir", default='save',
                       help='directory where model is stored')
    args = argparser.parse_args()
    do_filter(args.in_file, args.out_file, args.save_dir)

if __name__ == '__main__':
    main()

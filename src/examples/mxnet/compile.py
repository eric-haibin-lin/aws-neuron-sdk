# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation
"""
Export the BERT Model for Deployment
====================================

This script exports the BERT model to a hybrid model serialized as a symbol.json file,
which is suitable for deployment, or use with MXNet Module API.

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming- \
      Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

import argparse
import logging
import warnings
import os
import time

import mxnet as mx
import gluonnlp as nlp
from gluonnlp.model import get_model, BERTClassifier

nlp.utils.check_version('0.8.1')

parser = argparse.ArgumentParser(description='Export hybrid BERT base model.')

parser.add_argument('--model_parameters',
                    type=str,
                    default=None,
                    help='The model parameter file saved from training.')

parser.add_argument('--model_name',
                    type=str,
                    default='bert_12_768_12',
                    choices=['bert_12_768_12', 'bert_24_1024_16'],
                    help='BERT model name. Options are "bert_12_768_12" and "bert_24_1024_16"')

parser.add_argument('--dataset_name',
                    type=str,
                    default='book_corpus_wiki_en_uncased',
                    choices=['book_corpus_wiki_en_uncased', 'book_corpus_wiki_en_cased',
                             'wiki_multilingual_uncased', 'wiki_multilingual_cased',
                             'wiki_cn_cased'],
                    help='BERT dataset name. Options include '
                         '"book_corpus_wiki_en_uncased", "book_corpus_wiki_en_cased", '
                         '"wiki_multilingual_uncased", "wiki_multilingual_cased", '
                         '"wiki_cn_cased"')

parser.add_argument('--output_dir',
                    type=str,
                    default='./output_dir',
                    help='The directory where the exported model symbol will be created. '
                         'The default is ./output_dir')

parser.add_argument('--batch_size',
                    type=int,
                    default=1,
                    help='The batch size of inputs.')

parser.add_argument('--seq_length',
                    type=int,
                    default=128,
                    help='The maximum total input sequence length after WordPiece tokenization.'
                         'Sequences longer than this needs to be truncated, and sequences shorter '
                         'than this needs to be padded. Default is 384')

parser.add_argument('--dropout',
                    type=float,
                    default=0.1,
                    help='The dropout probability for the classification/regression head.')

parser.add_argument('--debug',
                    action='store_true',
                    help='Use imperative mode for debugging')

parser.add_argument('--no_length',
                    action='store_true',
                    help='Include valid length as inputs')

args = parser.parse_args()

# create output dir
output_dir = args.output_dir
nlp.utils.mkdir(output_dir)

###############################################################################
#                                Logging                                      #
###############################################################################

log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s',
                              datefmt='%H:%M:%S')
fh = logging.FileHandler(os.path.join(args.output_dir, 'hybrid_export_bert.log'), mode='w')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
log.addHandler(console)
log.addHandler(fh)
log.info(args)

###############################################################################
#                              Hybridize the model                            #
###############################################################################

seq_length = args.seq_length
bert, _ = get_model(
    name=args.model_name,
    dataset_name=args.dataset_name,
    pretrained=False,
    use_pooler=True,
    use_decoder=False,
    use_classifier=False)
net = BERTClassifier(bert, num_classes=2, dropout=args.dropout)

if args.model_parameters:
    net.load_parameters(args.model_parameters)
else:
    net.initialize()
    warnings.warn('--model_parameters is not provided. The parameter checkpoint (.params) '
                  'file will be created based on default parameter initialization.')
if not args.debug:
    net.hybridize(static_alloc=True, static_shape=True)

###############################################################################
#                            Prepare dummy input data                         #
###############################################################################

test_batch_size = args.batch_size

inputs = mx.nd.arange(test_batch_size * seq_length)
inputs = inputs.reshape(shape=(test_batch_size, seq_length))
token_types = mx.nd.zeros_like(inputs)
valid_length = mx.nd.arange(test_batch_size)
batch = inputs, token_types, valid_length


###############################################################################
#            Start Alternative Inferentia Compatible Implementation           #
###############################################################################


import math
f = mx.nd if args.debug else mx.sym

def broadcast_axis(data=None, axis=None, size=None, out=None, name=None, **kwargs):
    assert axis == 1
    ones = f.ones((1,size,1,1))
    out = f.broadcast_div(data, ones)
    return out

def div_sqrt_dim(data=None, out=None, name=None, **kwargs):
    assert '1024' in args.model_name or '768' in args.model_name
    units = 1024/16 if '1024' in args.model_name else 768/12
    return data / math.sqrt(units)

def embedding_op(data=None, weight=None, input_dim=None, output_dim=None, dtype=None,
                 sparse_grad=None, out=None, name=None, batch_mode=True, **kwargs):
    repeat = seq_length if batch_mode else test_batch_size * seq_length
    output_shape = (seq_length, output_dim) if batch_mode else (test_batch_size, seq_length, output_dim)
    #indices = data.reshape((-1))
    #x_idx = data.repeat(output_dim).reshape((1, -1))
    #y_idx = f.arange(output_dim, repeat=repeat).reshape((output_dim, -1)).transpose()
    #x_y = f.concat(x_idx, y_idx.reshape((1, -1)), dim=0)
    #encoded = f.gather_nd(weight, x_y)
    #encoded = f.broadcast_add(data.repeat(output_dim), weight.sum() * 0)
    encoded = f.broadcast_add(f.ones(output_shape), weight.sum() * 0 + data.sum() * 0)
    return encoded

def embedding(self, F, x, weight):
    out = embedding_op(x, weight, name='fwd', batch_mode=False, **self._kwargs)
    return out

def gelu(self, F, x):
    return 0.5 * x * (1 + F.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * (x ** 3))))

def layer_norm(self, F, data, gamma, beta):
    mean = data.mean(axis=self._axis, keepdims=True)
    delta = F.broadcast_sub(data, mean)
    var = (delta ** 2).mean(axis=self._axis, keepdims=True)
    X_hat = F.broadcast_div(delta, var.sqrt() + self._epsilon)
    return F.broadcast_add(F.broadcast_mul(gamma, X_hat), beta)

def arange_like(x, axis):
    if axis == 1:
        slice_x = x.slice(begin=(0, 0, 0), end=(1, None, 1)).reshape((-1))
    elif axis == 0:
        slice_x = x.slice(begin=(0, 0, 0), end=(None, 1, 1)).reshape((-1))
    else:
        raise NotImplementedError
    zeros = f.zeros_like(slice_x)
    if args.debug:
        arange = f.arange(start=0, repeat=1, step=1, stop=slice_x.shape[0], dtype='float32')
    else:
        arange = f.arange(start=0, repeat=1, step=1, stop=seq_length, dtype='float32')
    arange = f.elemwise_add(arange, zeros)
    return arange

def where(condition=None, x=None, y=None, name=None, attr=None, out=None, **kwargs):
    return x

nlp.model.GELU.hybrid_forward = gelu
mx.gluon.nn.LayerNorm.hybrid_forward = layer_norm
mx.gluon.nn.Embedding.hybrid_forward = embedding
f.contrib.arange_like = arange_like
f.Embedding = embedding_op
f.contrib.div_sqrt_dim = div_sqrt_dim
f.broadcast_axis = broadcast_axis
f.where = where

###############################################################################
#             End Alternative Inferentia Compatible Implementation            #
###############################################################################

def export(batch, prefix):
    """Export the model."""
    log.info('Exporting the model ... ')
    inputs, token_types, valid_length = batch
    out = net(inputs, token_types) if args.no_length else net(inputs, token_types, valid_length)
    if args.debug:
        exit()
    net.export(prefix, epoch=0)
    assert os.path.isfile(prefix + '-symbol.json')
    assert os.path.isfile(prefix + '-0000.params')

def infer(prefix):
    """Evaluate the model on a mini-batch."""
    log.info('Test inference with the model ... ')

    # import with SymbolBlock. Alternatively, you can use Module.load APIs.
    names = ['data0', 'data1', 'data2'] if not args.no_length else ['data0', 'data1']
    imported_net = mx.gluon.nn.SymbolBlock.imports(prefix + '-symbol.json',
                                                   names, prefix + '-0000.params')

    inputs = mx.nd.arange(test_batch_size * (seq_length))
    inputs = inputs.reshape(shape=(test_batch_size, seq_length))
    token_types = mx.nd.zeros_like(inputs)
    valid_length = mx.nd.arange(test_batch_size)

    # run forward inference
    out = imported_net(inputs, token_types) if args.no_length else imported_net(inputs, token_types, valid_length)
    mx.nd.waitall()

    # benchmark speed after warmup
    tic = time.time()
    num_trials = 10
    for _ in range(num_trials):
        out = imported_net(inputs, token_types) if args.no_length else imported_net(inputs, token_types, valid_length)
    mx.nd.waitall()
    toc = time.time()
    log.info('Batch size={}, Thoughput={:.2f} batches/s'
             .format(test_batch_size, num_trials / (toc - tic)))

def neuron_compile(prefix):
    # compile for Inferentia using Neuron
    if not args.no_length:
        inputs = {"data0" : mx.nd.ones(shape=(1, 128), name='data0'),
                  "data1" : mx.nd.ones(shape=(1, 128), name='data1'),
                  "data2" : mx.nd.ones(shape=(1,), name='data2')}
    else:
        inputs = {"data0" : mx.nd.ones(shape=(1, 128), name='data0'),
                  "data1" : mx.nd.ones(shape=(1, 128), name='data1')}

    sym, args_loaded, aux = mx.model.load_checkpoint(prefix, 0)
    sym, args_loaded, aux = mx.contrib.neuron.compile(sym, args_loaded, aux, inputs)

    # save compiled model
    mx.model.save_checkpoint(prefix + "_compiled", 0, sym, args_loaded, aux)


###############################################################################
#                              Export the model                               #
###############################################################################
if __name__ == '__main__':
    prefix = os.path.join(args.output_dir, 'classification')
    export(batch, prefix)
    infer(prefix)
    neuron_compile(prefix)

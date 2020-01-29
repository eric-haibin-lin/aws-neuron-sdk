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
    use_classifier=False,
    dropout=0.0)
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

inputs = mx.nd.arange(test_batch_size * seq_length * 768)
inputs = inputs.reshape(shape=(test_batch_size, seq_length, 768))
token_types = mx.nd.zeros_like(inputs)
position_embed = mx.nd.arange(seq_length * 768).reshape((seq_length, 768))
valid_length = mx.nd.arange(test_batch_size)
batch = inputs, token_types, position_embed, valid_length


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
    #encoded = f.broadcast_add(f.ones(output_shape), weight.sum() * 0 + data.sum() * 0)
    #return encoded
    return data

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
        slice_x = x.slice(begin=(0, 0, 0), end=(1, None, None)).reshape((-1, 768))
    #elif axis == 0:
    #    slice_x = x.slice(begin=(0, 0, 0), end=(None, 1, 1)).reshape((-1))
    else:
        raise NotImplementedError
    zeros = f.zeros_like(slice_x)
    # if args.debug:
    #     arange = f.arange(start=0, repeat=768, step=1, stop=slice_x.shape[0], dtype='float32')
    # else:
    #     arange = f.arange(start=0, repeat=768, step=1, stop=seq_length, dtype='float32')
    # arange = f.elemwise_add(arange, zeros)
    return zeros #arange

def where(condition=None, x=None, y=None, name=None, attr=None, out=None, **kwargs):
    return x

def dropout(data=None, p=None, mode=None, axes=None, cudnn_off=None, out=None, name=None, **kwargs):
    return data

def bert_model___call__(self, inputs, token_types, position_embed, valid_length=None, masked_positions=None):
    # pylint: disable=dangerous-default-value, arguments-differ
    """Generate the representation given the inputs.
    This is used in training or fine-tuning a BERT model.
    """
    return super(nlp.model.BERTModel, self).__call__(inputs, token_types, position_embed, valid_length, masked_positions)

def bert_model_hybrid_forward(self, F, inputs, token_types, position_embed,
                              valid_length=None, masked_positions=None):
    # pylint: disable=arguments-differ
    """Generate the representation given the inputs.
    This is used in training or fine-tuning a BERT model.
    """
    outputs = []
    seq_out, attention_out = self._encode_sequence(inputs, token_types, position_embed, valid_length)
    outputs.append(seq_out)

    if self.encoder._output_all_encodings:
        assert isinstance(seq_out, list)
        output = seq_out[-1]
    else:
        output = seq_out

    if attention_out:
        outputs.append(attention_out)

    if self._use_pooler:
        pooled_out = self._apply_pooling(output)
        outputs.append(pooled_out)
        if self._use_classifier:
            next_sentence_classifier_out = self.classifier(pooled_out)
            outputs.append(next_sentence_classifier_out)
    if self._use_decoder:
        assert masked_positions is not None, \
            'masked_positions tensor is required for decoding masked language model'
        decoder_out = self._decode(F, output, masked_positions)
        outputs.append(decoder_out)
    return tuple(outputs) if len(outputs) > 1 else outputs[0]

def bert_model__encode_sequence(self, inputs, token_types, position_embed, valid_length=None):
    """Generate the representation given the input sequences.
    This is used for pre-training or fine-tuning a BERT model.
    """
    # embedding
    embedding = self.word_embed(inputs)
    if self._use_token_type_embed:
        type_embedding = self.token_type_embed(token_types)
        embedding = embedding + type_embedding
    # encoding
    outputs, additional_outputs = self.encoder(embedding, position_embed, valid_length=valid_length)
    return outputs, additional_outputs

def bert_encoder___call__(self, inputs, position_embed, states=None, valid_length=None): #pylint: disable=arguments-differ
    """Encode the inputs given the states and valid sequence length.
    Parameters
    ----------
    inputs : NDArray or Symbol
        Input sequence. Shape (batch_size, length, C_in)
    states : list of NDArrays or Symbols
        Initial states. The list of initial states and masks
    valid_length : NDArray or Symbol
        Valid lengths of each sequence. This is usually used when part of sequence has
        been padded. Shape (batch_size,)
    Returns
    -------
    encoder_outputs: list
        Outputs of the encoder. Contains:
        - outputs of the transformer encoder. Shape (batch_size, length, C_out)
        - additional_outputs of all the transformer encoder
    """
    #return super(nlp.model.BERTEncoder, self).__call__(inputs, position_embed, states, valid_length)
    return mx.gluon.HybridBlock.__call__(self, inputs, position_embed, states, valid_length)

def bert_encoder_hybrid_forward(self, F, inputs, position_embed, states=None, valid_length=None, position_weight=None):
    # pylint: disable=arguments-differ
    """Encode the inputs given the states and valid sequence length.
    Parameters
    ----------
    inputs : NDArray or Symbol
        Input sequence. Shape (batch_size, length, C_in)
    states : list of NDArrays or Symbols
        Initial states. The list of initial states and masks
    valid_length : NDArray or Symbol
        Valid lengths of each sequence. This is usually used when part of sequence has
        been padded. Shape (batch_size,)
    Returns
    -------
    outputs : NDArray or Symbol, or List[NDArray] or List[Symbol]
        If output_all_encodings flag is False, then the output of the last encoder.
        If output_all_encodings flag is True, then the list of all outputs of all encoders.
        In both cases, shape of the tensor(s) is/are (batch_size, length, C_out)
    additional_outputs : list
        Either be an empty list or contains the attention weights in this step.
        The attention weights will have shape (batch_size, length, length) or
        (batch_size, num_heads, length, length)
    """
    # steps = F.contrib.arange_like(inputs, axis=1)
    mask = None
    # if valid_length is not None:
    #     ones = F.ones_like(steps)
    #     mask = F.broadcast_lesser(F.reshape(steps, shape=(1, -1)),
    #                               F.reshape(valid_length, shape=(-1, 1)))
    #     mask = F.broadcast_mul(F.expand_dims(mask, axis=1),
    #                            F.broadcast_mul(ones, F.reshape(ones, shape=(-1, 1))))
    #     if states is None:
    #         states = [mask]
    #     else:
    #         states.append(mask)
    # else:
    #     mask = None

    # if states is None:
    #     states = [steps]
    # else:
    #     states.append(steps)

    # positional encoding
    #positional_embed = F.Embedding(steps, position_weight, self._max_length, self._units)
    inputs = F.broadcast_add(inputs, F.expand_dims(position_embed, axis=0))
    #inputs = F.broadcast_add(inputs, position_embed)

    if self._dropout:
        inputs = self.dropout_layer(inputs)
    inputs = self.layer_norm(inputs)
    outputs = inputs

    all_encodings_outputs = []
    additional_outputs = []
    for cell in self.transformer_cells:
        outputs, attention_weights = cell(inputs, mask)
        inputs = outputs
        if self._output_all_encodings:
            if valid_length is not None:
                outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                         use_sequence_length=True, axis=1)
            all_encodings_outputs.append(outputs)

        if self._output_attention:
            additional_outputs.append(attention_weights)

    if valid_length is not None and not self._output_all_encodings:
        # if self._output_all_encodings, SequenceMask is already applied above
        outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                 use_sequence_length=True, axis=1)

    if self._output_all_encodings:
        return all_encodings_outputs, additional_outputs
    return outputs, additional_outputs

nlp.model.GELU.hybrid_forward = gelu
mx.gluon.nn.LayerNorm.hybrid_forward = layer_norm
mx.gluon.nn.Embedding.hybrid_forward = embedding
f.contrib.arange_like = arange_like
f.Embedding = embedding_op
f.contrib.div_sqrt_dim = div_sqrt_dim
f.broadcast_axis = broadcast_axis
f.where = where
f.Dropout = dropout
nlp.model.bert.BERTModel.__call__ = bert_model___call__
nlp.model.bert.BERTModel._encode_sequence = bert_model__encode_sequence
nlp.model.bert.BERTModel.hybrid_forward = bert_model_hybrid_forward
nlp.model.bert.BERTEncoder.__call__ = bert_encoder___call__
nlp.model.bert.BERTEncoder.hybrid_forward = bert_encoder_hybrid_forward

###############################################################################
#             End Alternative Inferentia Compatible Implementation            #
###############################################################################

def export(batch, prefix):
    """Export the model."""
    log.info('Exporting the model ... ')
    inputs, token_types, position_embed, valid_length = batch
    out = net(inputs, token_types, position_embed) if args.no_length else net(inputs, token_types, valid_length)
    if args.debug:
        exit()
    export_special(net, prefix, epoch=0)
    assert os.path.isfile(prefix + '-symbol.json')
    assert os.path.isfile(prefix + '-0000.params')

def export_special(net, path, epoch):
    sym = net._cached_graph[1]
    sym.save('%s-symbol.json'%path, remove_amp_cast=False)

    arg_names = set(sym.list_arguments())
    aux_names = set(sym.list_auxiliary_states())
    arg_dict = {}
    save_fn = mx.nd.save
    embedding_dict = {}
    for name, param in net.collect_params().items():
        if 'position_weight' in name or 'word_embed_embedding0_weight' in name or 'token_type_embed_embedding0_weight' in name:
            embedding_dict[name] = param._reduce()
        elif name in arg_names:
            arg_dict['arg:%s'%name] = param._reduce()
        else:
            assert name in aux_names, name
            arg_dict['aux:%s'%name] = param._reduce()
    save_fn('%s-%04d.params'%(path, epoch), arg_dict)
    save_fn('%s-%04d.embeddings'%(path, epoch), embedding_dict)

def infer(prefix):
    """Evaluate the model on a mini-batch."""
    log.info('Test inference with the model ... ')

    # import with SymbolBlock. Alternatively, you can use Module.load APIs.
    names = ['data0', 'data1', 'data2']
    imported_net = mx.gluon.nn.SymbolBlock.imports(prefix + '-symbol.json',
                                                   names, prefix + '-0000.params')

    inputs = mx.nd.arange(test_batch_size * seq_length * 768)
    inputs = inputs.reshape(shape=(test_batch_size, seq_length, 768))
    token_types = mx.nd.zeros_like(inputs)
    valid_length = mx.nd.arange(test_batch_size)

    # run forward inference
    out = imported_net(inputs, token_types, position_embed) if args.no_length else imported_net(inputs, token_types, position_embed, valid_length)
    mx.nd.waitall()

    # benchmark speed after warmup
    tic = time.time()
    num_trials = 10
    for _ in range(num_trials):
        out = imported_net(inputs, token_types, position_embed) if args.no_length else imported_net(inputs, token_types, position_embed, valid_length)
    mx.nd.waitall()
    toc = time.time()
    log.info('Batch size={}, Thoughput={:.2f} batches/s'
             .format(test_batch_size, num_trials / (toc - tic)))

def neuron_compile(prefix):
    # compile for Inferentia using Neuron
    if not args.no_length:
        assert False
        inputs = {"data0" : mx.nd.ones(shape=(test_batch_size, seq_length), name='data0'),
                  "data1" : mx.nd.ones(shape=(test_batch_size, seq_length), name='data1'),
                  "data2" : mx.nd.ones(shape=(test_batch_size,), name='data2')}
    else:
        inputs = {"data0" : mx.nd.ones(shape=(test_batch_size, seq_length, 768), name='data0'),
                  "data1" : mx.nd.ones(shape=(test_batch_size, seq_length, 768), name='data1'),
                  "data2" : mx.nd.ones(shape=(test_batch_size, seq_length), name='data2')}

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
    # neuron_compile(prefix)

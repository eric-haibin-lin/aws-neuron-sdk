import mxnet as mx
import gluonnlp as nlp
import argparse

parser = argparse.ArgumentParser(description='Export hybrid BERT base model.')

parser.add_argument('--compiled_model',
                    type=str,
                    default=None,
                    help='The compiled neuron model.')
parser.add_argument('--max_len',
                    type=int,
                    default=128,
                    help='The maximum length')

args = parser.parse_args()

sentence = 'Neuron is awesome'
sentence = 'Neuron is great'
sentence = 'Neuron is confusing'
max_len = args.max_len
_, vocabulary = nlp.model.get_model('bert_12_768_12',
                                    dataset_name='book_corpus_wiki_en_uncased',
                                    pretrained=False)
tokenizer = nlp.data.BERTTokenizer(vocabulary)
transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=max_len,
                                           pair=False, pad=True)

prefix = args.compiled_model + '-' + str(max_len)
embedding_dict = mx.nd.load(prefix + '-0000.embeddings')
sym, args, aux = mx.model.load_checkpoint(prefix, 0)
inputs, seq_len, token_types = transform([sentence])

inputs_arr = mx.nd.array([inputs])
token_types_arr = mx.nd.array([token_types])
postional_arr = mx.nd.arange(max_len)

args['data0'] = mx.nd.take(embedding_dict['bertmodel0_word_embed_embedding0_weight'], inputs_arr)
args['data1'] = mx.nd.take(embedding_dict['bertmodel0_token_type_embed_embedding0_weight'], token_types_arr)
args['data2'] = mx.nd.take(embedding_dict['bertencoder0_position_weight'], postional_arr)
args['data3'] = mx.nd.array([seq_len])

# TODO use neuron context
print(args['data0'].shape)
print(args['data1'].shape)
print(args['data2'].shape)
print(args['data3'].shape)

exe = sym.bind(ctx=mx.cpu(), args=args, aux_states=aux, grad_req='null')
exe.forward(data0=args['data0'], data1=args['data1'], data2=args['data2'])
out = exe.outputs[0]
print(exe.outputs[0])
label = mx.nd.argmax(out, axis=1)
for l in label:
    print('positive' if l.asscalar() == 1 else 'negative')

import mxnet as mx
import gluonnlp as nlp
import argparse

parser = argparse.ArgumentParser(description='Export hybrid BERT base model.')

parser.add_argument('--compiled_model',
                    type=str,
                    default=None,
                    help='The compiled neuron model.')
args = parser.parse_args()

sentence = 'Neuron is awesome'
_, vocabulary = nlp.model.get_model('bert_12_768_12',
                                    dataset_name='book_corpus_wiki_en_uncased',
                                    pretrained=False)
tokenizer = nlp.data.BERTTokenizer(vocabulary)
transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=128,
                                           pair=False, pad=False)

sym, args, aux = mx.model.load_checkpoint(args.compiled_model, 0)
inputs, seq_len, token_types = transform([sentence])
args['data0'] = mx.nd.array([inputs]).repeat(768).reshape((1, -1, 768))
args['data1'] = mx.nd.array([token_types]).repeat(768).reshape((1, -1, 768))
#args['data2'] = mx.nd.array([seq_len])

# TODO use neuron context
exe = sym.bind(ctx=mx.cpu(), args=args, aux_states=aux, grad_req='null')
exe.forward(data0=args['data0'], data1=args['data1'])#, data2=args['data2'])
out = exe.outputs[0]
label = mx.nd.argmax(out, axis=1)
print('positive' if label.asscalar() == 1 else 'negative')

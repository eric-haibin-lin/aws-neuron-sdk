import mxnet as mx
import gluonnlp as nlp
import argparse

parser = argparse.ArgumentParser(description='Run inference with compiled BERT model.')
parser.add_argument('--compiled_model', type=str,
                    required=True, help='The compiled neuron model.')
parser.add_argument('--seq_length', type=int,
                    default=128, help='The maximum total input sequence length.')
args = parser.parse_args()

# TODO: we assume batch_size = 1 in this demo
sentence = 'neuron compiler is awesome!'
bert_model, vocabulary = nlp.model.get_model('bert_12_768_12',
                                    dataset_name='book_corpus_wiki_en_uncased',
                                    pretrained=False,
                                    use_decoder=False,
                                    use_classifier=False)
tokenizer = nlp.data.BERTTokenizer(vocabulary)
transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=args.seq_length, pair=False)
inputs, seq_len, token_types = transform([sentence])

sym, args, aux = mx.model.load_checkpoint(args.compiled_model, 0)
args['data0'] = mx.nd.array([inputs])
args['data1'] = mx.nd.array([token_types])
args['data2'] = mx.nd.array([seq_len])

# TODO use neuron context
exe = sym.bind(ctx=mx.cpu(), args=args, aux_states=aux, grad_req='null')
exe.forward(data0=args['data0'], data1=args['data1'], data2=args['data2'])
out = exe.outputs[0]
label = mx.nd.argmax(out, axis=1)
print('positive' if label.asscalar() == 1 else 'negative')

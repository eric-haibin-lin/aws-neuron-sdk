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
sentence = 'Neuron is great'
sentence = 'Neuron is confusing'
_, vocabulary = nlp.model.get_model('bert_12_768_12',
                                    dataset_name='book_corpus_wiki_en_uncased',
                                    pretrained=False)
tokenizer = nlp.data.BERTTokenizer(vocabulary)
transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=128,
                                           pair=False, pad=False)

embedding_dict = mx.nd.load(args.compiled_model + '-0000.embeddings')
sym, args, aux = mx.model.load_checkpoint(args.compiled_model, 0)
inputs, seq_len, token_types = transform([sentence])

inputs_arr = mx.nd.array([inputs])
token_types_arr = mx.nd.array([token_types])
postional_arr = mx.nd.arange(len(inputs))

args['data0'] = mx.nd.take(embedding_dict['bertmodel0_word_embed_embedding0_weight'], inputs_arr)
args['data1'] = mx.nd.take(embedding_dict['bertmodel0_token_type_embed_embedding0_weight'], token_types_arr)
args['data2'] = mx.nd.take(embedding_dict['bertencoder0_position_weight'], postional_arr)

# TODO use neuron context
exe = sym.bind(ctx=mx.cpu(), args=args, aux_states=aux, grad_req='null')
exe.forward(data0=args['data0'], data1=args['data1'], data2=args['data2'])
out = exe.outputs[0]
label = mx.nd.argmax(out, axis=1)
print('positive' if label.asscalar() == 1 else 'negative')

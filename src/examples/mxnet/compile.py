import mxnet as mx
import numpy as np

sym, args, aux = mx.model.load_checkpoint('dir/classification', 0)

# compile for Inferentia using Neuron
inputs = {"data0" : mx.nd.ones(shape=(1, 128), name='data0'),
          "data1" : mx.nd.ones(shape=(1, 128), name='data1'),
          "data2" : mx.nd.ones(shape=(1,), name='data2')}

sym, args, aux = mx.contrib.neuron.compile(sym, args, aux, inputs)

# save compiled model
# mx.model.save_checkpoint("bert_compiled", 0, sym, args, aux)

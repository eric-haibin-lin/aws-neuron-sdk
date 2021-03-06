# BERT demo for AWS Neuron

To enable a performant BERT model on Inferentia, we must use a Neuron compatible BERT implementation. This demo shows a Neuron compatible BERT Large implementation that is functionally equivalent to open source BERT Large. We then show how to compile and run it on an Inf1.2xlarge instance. This demo uses Tensorflow-Neuron, BERT Large weights fine tuned for MRPC and also shows the performance achieved by the Inf1 instance. 

## Table of Contents

1. Launch EC2 instances 
2. Compiling Neuron compatible BERT Large for Inferentia
   * Create saved model from open source BERT Large
   * Compile model using Neuron compatible BERT Large for Inferentia
3. Running the inference demo
   * Launching the BERT demo server
   * Sending requests to server from multiple clients

## Launch EC2 instances

For this demo, we will use a c5.4xlarge EC2 instance for compiling the BERT Model and an Inf1.2xlarge instance for running the demo itself. For both of these instances choose the latest an Ubuntu 18 Deep Learning AMI (DLAMI). After starting the instance please make sure you update the Neuron software to the latest version before continuing with this demo. See the [DLAMI Release Notes](https://github.com/aws/aws-neuron-sdk/blob/master/release-notes/dlami-release-notes.md#base-and-conda-dlami-on-ubuntu) for more information on updating Neuron software.

## Compiling Neuron compatible BERT Large for Inferentia
Connect to your c5.4xlarge instance and update tensorflow-neuron and neuron-cc

```bash
source activate aws_neuron_tensorflow_p36

conda install numpy=1.17.2 --yes --quiet
conda update tensorflow-neuron
```


### Create saved model from open source BERT Large
We used publicly available instructions to generate a saved model for open source BERT Large using fine-tuned MRPC weights. The steps to generate this model can be found [here](https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks). Save the fine-tuned MRPC model as a saved model as described [here](https://github.com/google-research/bert/issues/146#issuecomment-441168098).


Place the saved model in a directory named "bert-saved-model" under the bert_demo directory and proceed to the next section.

### Compile model using Neuron compatible BERT Large for Inferentia
In the same conda environment and directory containing your bert-saved-model-neuron and bert_demo scripts, run the following :

```bash
python bert_model.py --input_saved_model ./bert-saved-model/ --output_saved_model ./bert-saved-model-neuron --crude_gelu
```


This compiles BERT large for an input size of 128 and batch size of 4. The compilation output is stored in bert_saved_model_neuron. Move this to your Inf1 instances for inferencing. The bert_model.py script encapsulates all the steps necessary for this. For details on what is done by bert_model.py please refer to Appendix 1.



## Running the inference demo
Connect to your Inf1.2xlarge instance and update tensorflow-neuron, aws-neuron-runtime and aws-neuron-tools :

```bash

source activate aws_neuron_tensorflow_p36
conda install numpy=1.17.2 --yes --quiet
conda update tensorflow-neuron

```

### Launching the BERT demo server
Copy the compiled model (bert_saved_model_neuron) from your c5.4xlarge to your Inf1.2xlarge instance. Place the model in the same directory as the bert_demo scripts. Then from the same conda environment launch the BERT demo server :

```bash
python bert_server.py --dir bert-saved-model-neuron --parallel 4
```


This loads 4 BERT Large models, one into each of the 4 NeuronCores in a single Inferentia device. For each of the 4 models, the BERT demo server opportunistically stiches together asynchronous requests into batch 4 requests. When there are insufficient pending requests, the server creates dummy requests for batching.

Wait for the bert_server to finish loading the BERT models to Inferentia memory. When it is ready to accept requests it will print the inferences per second once every second. This reflects the number of real inferences only. Dummy requests created for batching are not credited to inferentia performance.

### Sending requests to server from multiple clients
On the same Inf1.2xlarge instance, launch a separate linux terminal. From the bert_demo directory execute the following commands :


```bash
source activate aws_neuron_tensorflow_p36
for i in {1..48}; do python bert_client.py --cycle 128 & done
```

This spins up 48 clients, each of which sends 128 inference requests. The expected performance is about 200 inferences/second for a single instance of inf1.2xlarge.


## Appendix 1 :
The bert_model.py script does a few things :
1. Define a Neuron compatible implementation of BERT Large. For inference, this is functionally equivalent to the open source BERT Large
2. Extract BERT Large weights from the saved model pointed to by --input_saved_model and associates it with the Neuron compatible implementation
3. Invoke Tensorflow-Neuron compile to compile this model for Inferentia using the newly associated weights
4. Finally, the compiled model is saved into the location given by --output_saved_model

The changes needed for a Neuron compatible BERT implementation is given in Appendix 2.


## Appendix 2 :
The Neuron compatible implementation of BERT Large is functionally equivalent to the open source version when used for inference. However, the detailed implementation does differ and here are the list of changes :

1. Data Type Casting : If the original BERT an FP32 model, bert_model.py contains manually defined cast operators to enable mixed-precision. FP16 is used for multi-head attention and fully-connected layers, and fp32 everywhere else. This will be automated in a future release.
2. Remove Unused Operators: A model typically contains training operators that are not used in inference, including a subset of the reshape operators. Those operators do not affect inference functionality and have been removed.
3. Reimplementation of Selected Operators : A number of operators (mainly mask operators), has been reimplemented to bypass a known compiler issue. This will be fixed in a planned future release. 
4. Manually Partition Embedding Ops to CPU : The embedding portion of BERT has been partitioned manually to a subgraph that is executed on the host CPU, without noticable performance impact. In near future, we plan to implement this through compiler auto-partitioning without the need for user intervention.

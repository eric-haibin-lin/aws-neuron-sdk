# BERT demo for AWS Neuron

Table of Contents
* Launch EC2 instances
* Compiling Neuron compatible BERT for Inferentia
* Running the inference demo

## Launch EC2 instances

For this demo, we will use an inf1.2xlarge EC2 instance for compiling the BERT model and running the demo itself. For both of these instances choose the latest an Ubuntu 18 Deep Learning AMI (DLAMI). After starting the instance please make sure you update the Neuron software to the latest version before continuing with this demo. See the DLAMI Release Notes (https://github.com/aws/aws-neuron-sdk/blob/master/release-notes/dlami-release-notes.md#base-and-conda-dlami-on-ubuntu) for more information on updating Neuron software.

## Compiling Neuron compatible BERT for Inferentia

Connect to your instance and update `mxnet-neuron` and `neuron-cc`

```bash
source activate aws_neuron_mxnet_p36

conda install numpy=1.17.2 --yes --quiet
conda update mxnet-neuron
pip install https://github.com/dmlc/gluon-nlp/tarball/v0.9.x
```

We used publicly available instructions to generate a saved model for open source BERT using fine-tuned SST-2 weights.
The steps to generate this model can be found [here](https://gluon-nlp.mxnet.io/v0.9.x/model_zoo/bert/index.html#sentence-classification),
or you can download a trained model on SST-2 [here](https://dist-bert.s3.amazonaws.com/demo/finetune/sst.params).
Place the saved model in a directory named "gluonnlp_bert" under the `bert_demo` directory.

Run the following to compile BERT for an input size of 128 and batch size of 1.
The compilation output is stored in `neuron_bert`.
```
python compile.py --output_dir neuron_bert --seq_length 128  --batch_size 1 --model_parameters gluonnlp_bert/sst.params --model_name bert_12_768_12
```

## Running the inference demo

Run the command below to run BERT inference on CPU:
```bash
python inference.py --compiled_model neuron_bert/classification-bert_12_768_12 --max_len 128
```

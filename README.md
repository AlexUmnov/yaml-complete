# yaml-complete
A simple POC for yaml code completion 


If imports refuse to work, use 

`export PYTHONPATH="."`

First install dependencies:

`conda create env -f environment.yaml`
`conda activate yaml-complete`

To run local models and finetuning of the models, install pytorch (choose appropriate version):

`conda install pytorch  pytorch-cuda=11.8 -c pytorch -c nvidia`
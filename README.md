# yaml-complete
A simple POC for yaml code completion 


If imports refuse to work, use 

`export PYTHONPATH="."`

First install dependencies:

`conda create env -f environment.yaml`
`conda activate yaml-complete`

To run local models install pytorch (choose appropriate version):

`conda install pytorch  pytorch-cuda=12.1 -c pytorch -c nvidia`

To run OpenAI models please put your key in `.open-ai-api-key`

To gather data use scripts from `data_gathering`, this requires your github token in `.github-token`

The order is the following:
- `get_github_files.py`
- `train_test_split.py` to split into two folders, also fitlers the raw data
- `create_evaluations.py` to create a test set (can also be used to create training data for finetunning)

You can run evaluations by running 
`python evaluation/evaluate.py --test_file evaluation_data.json --output_file evaluation_metrics.csv`

But be avarage that you need to generate data first and have a gpu which can handle all the models. 

Otherwise you can limit models to evaluate with `--include-predictors pred1,pred2,etc.`

To run the local damo in Tkinter gui you can run `python gui.py -p pred_name`

This project also includes a finetuned model, which you can find here 
- https://huggingface.co/alexvumnov/yaml_completion
- https://huggingface.co/alexvumnov/yaml_completion_8_bit

Finetuning process can be viewed at `notebooks/2.yaml_finetune.ipynb`
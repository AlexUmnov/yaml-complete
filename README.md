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

Stats for the models tested:

|    | name                              | exact_match          | iou                 | legit_yaml         | average_latency (s) | average_cost ($)       |
|----|-----------------------------------|----------------------|---------------------|--------------------|---------------------|------------------------|
| 0  | openai_gpt_4o_mini_chat_predictor | 0.09978768577494693  | 0.25211440413702635 | 0.6645435244161358 | 0.5924776577392707  | 3.863885350318473e-05  |
| 1  | openai_gpt_4o_chat_predictor      | 0.12951167728237792  | 0.28123878199463853 | 0.6985138004246284 | 0.5847456784258476  | 0.0012565711252653937  |
| 2  | openai_gpt_3_5_instruct_predictor | 0.027600849256900213 | 0.10787349460881061 | 0.5456475583864119 | 0.6200108781235486  | 0.00037238747346072235 |
| 3  | stable_code_hf                    | 0.12951167728237792  | 0.19394219136990798 | 0.8089171974522293 | 0.24507619772747063 | 0.0                    |
| 4  | codegen_350_hf                    | 0.014861995753715499 | 0.06384083412811624 | 0.445859872611465  | 0.2998334885655948  | 0.0                    |
| 5  | codegen_350_mono_hf               | 0.012738853503184714 | 0.06143149847763767 | 0.4607218683651805 | 0.3199998213733584  | 0.0                    |
| 6  | deepseek_coder_v2_lite_base       | 0.15286624203821655  | 0.23311644761698475 | 0.89171974522293   | 0.8314042830416605  | 0.0                    |
| 7  | code_llama                        | 0.15286624203821655  | 0.2370654605983271  | 0.9044585987261147 | 0.3316897293058424  | 0.0                    |
| 8  | codegen2_16_b_p                   | 0.012738853503184714 | 0.060622706571818   | 0.4394904458598726 | 1.1037331518347349  | 0.0                    |
| 9  | yaml_complete_finetuned           | 0.14861995753715498  | 0.28872705647423663 | 0.8259023354564756 | 0.21932720429325306 | 0.0                    |
| 10 | yaml_complete_finetuned_8_bit     | 0.14861995753715498  | 0.29205720302416377 | 0.8301486199575372 | 0.592521419444155   | 0.0                    |
| 11 | yaml_complete_finetuned_cpu       | 0.14861995753715498  | 0.28872705647423663 | 0.8259023354564756 | 0.7195762605930337  | 0.0                    |


But be avarage that you need to generate data first and have a gpu which can handle all the models. 

Otherwise you can limit models to evaluate with `--include-predictors pred1,pred2,etc.`

To run the local damo in Tkinter gui you can run `python gui.py -p pred_name`

Completions are generated after pressing <TAB>

This project also includes a finetuned model, which you can find here 
- https://huggingface.co/alexvumnov/yaml_completion
- https://huggingface.co/alexvumnov/yaml_completion_8_bit

Finetuning process can be viewed at `notebooks/2.yaml_finetune.ipynb`
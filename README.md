# Explaining Offensive Language Detection in Context
The goal of this project was that of explaining a RoBERTa Hate Speech classifier fine-tuned with contextual information (the previous post) and compare the rationales extracted with LIME and SHAP with those obtained by the same architecture fine-tuned only on the target posts (no context was provided to the model in this case). More details can be found [here](https://drive.google.com/file/d/1hgns8Z0Pkrq7NPgJqKsNwohh79zrkG_C/view?usp=sharing).

---
## Setup
Install requirements 
```
pip install -r requirements.txt
```
## Run fine-tuning
You can choose your preferred random seed. Use `--context` if you with to fine-tune the dataset with contextual information.
```
python run_finetuning.py --dataset_file_path=datasets/yu22/data/dataset.jsonl --random_seed=349
```
## Evaluate
Use the same seed used during fine-tuning.  
Use `--context` if you with to consider contextual information.  
Define with `--checkpoint_path` the location of the model weights.
```
python run_eval.py --dataset_file_path=datasets/yu22/data/dataset.jsonl --random_seed=349 --checkpoint_path=/path/model/weights
```
## Get Explanations
Also here, make sure to use the same seed used during fine-tuning.  
Same considerations as for the model evaluation regarding `--context` and `--checkpoint_path`.
### LIME

```
python run_lime.py --dataset_file_path=datasets/yu22/data/dataset.jsonl --checkpoint_path=/path/weight/model/to/explain --random_seed=349
```
### SHAP
```
python run_shap.py --dataset_file_path=datasets/yu22/data/dataset.jsonl --checkpoint_path=/path/weight/model/to/explain --random_seed=349
```

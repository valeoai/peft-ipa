<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Finetuning LLaMA on commonsense reasoning tasks using IPA

This directory includes the IPA implementation.

## Setup
1. Install dependencies
```bash
conda create -n ipa python=3.10
conda activate ipa
pip install -r requirements.txt
```

## Datasets
1. Download the complete commonsense datasets from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset) and download the commonsense 170k finetuning dataset from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json), then organize the data as follows
```bash
# Store the complete commonsense datasets
./dataset
# rest of the files
./experiment
# Finetuning commonsense dataset
./commonsense_170k.json
...
```

## Finetuning and Evaluation

### Finetuning (`./llama_7B_IPA.sh`)
This file contains the code to finetune LLaMA-7B using IPA. User can specify different IPA configuration for finetuning. To be specific, the arguments are following:
1. rank r, 
2. scaling factor of projected features
3. smoothing factor lambda (0.0 if using only the proj of the first batch)
4. learning rate
5. the destination for saving the fine-tuned model 
6. cuda device number.
 
An example could be:
```
sh llama_7B_IPA.sh 32 0.25 1e-4 1e-4 ./finetuned_result/ipa_r32 0
```

### Evaluation

This file contains the code to evaluate LLaMA-7B finetuned with IPA on the eight commonsense reasoning tasks. The first argument is the address of the IPA weight, the second argument specifies where you would like to save the evaluation result, and the last argument determines which GPU to use.

An example could be:
```
sh llama_7B_IPA_eval.sh ./finetuned_result/ipa_r32 0
```



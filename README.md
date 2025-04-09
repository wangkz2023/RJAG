# RJAG: Retrieval Judgment Augmented Generation

## Overview
Large Language Models (LLMs) inevitably suffer from hallucinations, 
as relying solely on their parametric knowledge cannot guarantee the accuracy of generated content. 
To enhance text generation, retrieval-augmented generation (RAG) is proposed to incorporate external knowledge to achieve this. 
However, its effectiveness heavily depends on the relevance of retrieved documents, which poses a critical challenge: 
how to ensure the accuracy and reliability of model responses when retrieval results are inaccurate. 
Tackling this challenge, **we propose Retrieval Judgment Augmented Generation (RJAG)**, 
a method that can enhance RAG through LLM-driven fine-grained relevance judgment and a task-adaptive knowledge combination strategy. 
RJAG judges and dynamically combines retrieved documents for both open-ended generation and closed-ended selection tasks. 
Additionally, large-scale web search is also included to expand the knowledge beyond static corpora. 
Experimental results on multiple benchmarks show that RJAG outperforms existing RAG methods, 
which will significantly enhance the accuracy and reliability while maintaining the system's simplicity.

<div align="center">
  <img src="https://github.com/wangkz2023/RJAG/blob/main/img/figure1.png" width="80%">

  
  <img src="https://github.com/wangkz2023/RJAG/blob/main/img/RJAG.png" width="100%">
</div>

## Requirements
**Note: We use Python 3.11 for RJAG** To get started, install conda and run:
```
git clone https://github.com/wangkz2023/RJAG.git
conda create -n RJAG python=3.11
...
pip install -r requirements.txt
```

## Download
- Download the **eval_data** created by [Self-RAG (Asai et al., 2023)](https://github.com/AkariAsai/self-rag) on PopQA, PubQA, Bio and Arc_challenge with retrieved results
- Download the **data** from [RJAG-data](https://drive.google.com/drive/folders/1E-n1p4r0VLSKL8a0su3PRGtX6_dtqtSF?usp=sharing)

## Data Preprocess
Run the following command to preprocess the dataset for questions and retrieval results. Specifically for PopQA, the label of each (question, passage) pair is also collected.
```
bash run_data_preprocess.sh
```

## Run RJAG
### Web Knowledge Preparation
Run the following command to gather web knowledge for inference.
```
bash run_knowledge_preparation.sh
```

### Relevance Judgment
Use a large language model to judge the relevance of retrieved documents
```
python llm-as-a-judge/llm-as-a-judge.py
```

### Inference
#### RJAG
Run the following command for RJAG inference.
```
bash rjag.sh
```

### Evaluation
For Bio evaluation, please follow the instructions at the [FactScore (Min et al., 2023)](https://github.com/shmsw25/FActScore) official repository. 
```
python -m factscore.factscorer --data_path YOUR_OUTPUT_FILE  --model_name retrieval+ChatGPT --cache_dir YOUR_CACHE_DIR --openai_key YOUR_OPEN_AI_KEY --verbose
```

It is worth mentioning that, previous FactScore adopted **text-davinci-003** by default, which has been [deprecated since 2024-01-04](https://platform.openai.com/docs/deprecations) and replaced by **gpt-3.5-turbo-instruct**.
Both results reported are based on the **gpt-3.5-turbo-instruct**, which may different from the results of the original author's paper CRAG.

For the other datasets, run the following command.
```
bash run_deepseek_judge_eval.sh
```


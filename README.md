# RJAG: Retrieval Judgment Augmented Generation

## Overview
Large Language Models (LLMs) suffer from hallucinations due to their reliance on parametric knowledge, limiting content accuracy. Retrieval-Augmented Generation (RAG) enhances generation by incorporating external knowledge, but its effectiveness depends on retrieval relevance, posing a key challenge: how to ensure robust generation when retrieval is unreliable?
To address this, we propose the **Retrieval Judgment Augmented Generation (RJAG)**, which enhances RAG through LLM-driven fine-grained relevance judgment and a task-adaptive knowledge combination strategy. RJAG judges and dynamically combines retrieved documents for both open-ended generation and closed-ended selection tasks, improving response quality. Furthermore, large-scale web search is incorporated to expand knowledge beyond static corpora.
Experiments on multiple benchmarks show that RJAG outperforms existing RAG methods, significantly enhancing accuracy and reliability while maintaining system simplicity.
<div align=center>
  <img src="https://github.com/wangkz2023/RJAG/blob/main/img/figure1.png" width=60%>
  <img src="https://github.com/wangkz2023/RJAG/blob/main/img/RJAG.png" width=60%>
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


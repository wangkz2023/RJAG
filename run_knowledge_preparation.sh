#!/bin/sh
cd scripts
dataset=popqa
OPENAI_KEY=
SEARCH_KEY=

#python web_knowledge_llm-as-a-judge.py \
#--input_queries ../data/$dataset/sources \
#--openai_key $OPENAI_KEY \
#--search_key $SEARCH_KEY \
#--task $dataset --mode wiki\
#--output_file ../data/$dataset/web_search

python web_knowledge_preparation.py \
--model_path ../model/finetuned_t5_evaluator \
--input_queries ../data/$dataset/sources \
--openai_key $OPENAI_KEY \
--search_key $SEARCH_KEY \
--task $dataset --mode wiki\
--output_file ../data/$dataset/web_search



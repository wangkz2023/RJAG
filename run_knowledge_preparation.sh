#!/bin/sh
cd scripts
dataset=popqa
OPENAI_KEY=
SEARCH_KEY=

python web_knowledge.py \
--input_queries ../data/$dataset/sources \
--openai_key $OPENAI_KEY \
--search_key $SEARCH_KEY \
--task $dataset --mode wiki\
--output_file ../data/$dataset/web_search


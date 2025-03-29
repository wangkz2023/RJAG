#!/bin/sh

cd scripts

dataset=popqa
python deepseek_judge_inference_H+S_orWeb.py \
--generator_path YOUR_GENERATOR_PATH \
--input_file ../data/$dataset/popqa_deepseek_as_a_judge_all.txt \
--output_file ../data/$dataset/output/deepseek_judge_popqa_inference_H+S_orWeb_100 \
--web_knowledge_path ../data/$dataset/web_search \
--task $dataset \
--device cuda:0 \
--ndocs 10 --batch_size 8

#!/bin/sh

cd scripts

dataset=popqa
python deepseek_judge_inference_H_or_Web+S.py \
--generator_path YOUR_GENERATOR_PATH \
--input_file ../data/$dataset/popqa_deepseek_as_a_judge_all.txt \
--output_file ../data/$dataset/output/deepseek_judge_popqa_inference_H_or_Web+S_100 \
--web_knowledge_path ../data/$dataset/web_search \
--task $dataset \
--device cuda:0 \
--ndocs 10 --batch_size 8

dataset=bio
python deepseek_judge_inference_H_or_Web+S.py \
--generator_path YOUR_GENERATOR_PATH \
--input_file ../data/$dataset/bio_deepseek_as_a_judge_all.txt \
--output_file ../data/$dataset/output/deepseek_judge_bio_inference_H_or_Web+S_100 \
--web_knowledge_path ../data/$dataset/web_search \
--task $dataset \
--device cuda:0 \
--ndocs 10 --batch_size 8

dataset=pubqa
python deepseek_judge_inference_H.py \
--generator_path YOUR_GENERATOR_PATH \
--input_file ../data/$dataset/pubqa_deepseek_as_a_judge_all.txt \
--output_file ../data/$dataset/output/deepseek_judge_pubqa_inference_H_100 \
--task $dataset \
--device cuda:0 \
--ndocs 10 --batch_size 8

dataset=arc_challenge
python deepseek_judge_inference_H.py \
--generator_path YOUR_GENERATOR_PATH \
--input_file ../data/$dataset/arc_challenge_deepseek_as_a_judge_all.txt \
--output_file ../data/$dataset/output/deepseek_judge_arc_challenge_inference_H_100 \
--task $dataset \
--device cuda:0 \
--ndocs 10 --batch_size 8
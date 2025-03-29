cd scripts

#python eval.py \
#  --input_file eval_data/popqa_longtail_w_gs.jsonl \
#  --eval_file ../data/popqa/output/deepseek_judge_popqa_inference_H_100 \
#  --metric match

#python eval.py \
#  --input_file ../eval_data/health_claims_processed.jsonl \
#  --eval_file ../data/pubqa/output/deepseek_judge_pubqa_inference_H_100 \
#  --metric match --task fever

python eval.py \
  --input_file eval_data/arc_challenge_processed.jsonl \
  --eval_file ../data/arc_challenge/output/deepseek_judge_arc_challenge_inference_H_100 \
  --metric match --task arc_c
#run simulation using 5 student user simulator

PYTHONPATH=scripts python scripts/score_best_articles_llmsim.py \
  --input data/judged_teacher_news.jsonl \
  --out data/judged_teacher_news_llmsim_s2.jsonl \
  --n_simulated_readers 5 \
  --pre_cache cache/pre_cache_train.jsonl

#split the dataset
python scripts/split_jsonl.py \
  --input data/judged_teacher_news_llmsim_s2.jsonl \
  --train data/train_llmsim_s2.jsonl \
  --dev data/dev_llmsim_s2.jsonl \
  --test data/test_llmsim_s2.jsonl \
  --test_size 300 \
  --dev_size 150


# build the dataset for SFT training, using top 75% of the articles, and including the QAs in the input
python scripts/build_kg_filtered_sft.py \
  --input data/train_llmsim_s2.jsonl \
  --out data/sft_kg_filtered_top75.jsonl \
  --top_percent 0.75 \
  --use_qas


# build a baseline:
python scripts/build_kg_filtered_sft.py \
  --input data/train_llmsim_s2.jsonl \
  --out data/sft_all.jsonl \
  --min_kg -999 \
  --use_qas

# then the evaluation scripts will compare the different settings

# sft all training
python scripts/train_sft_news.py \
  --data data/sft_all.jsonl \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --out outputs/sft_all
  

#KG-filtered SFT
python scripts/train_sft_news.py \
  --data data/sft_kg_filtered_top75.jsonl \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --out outputs/sft_kg_filtered_top75


# Since we already have four candidates per example, we can make cheap DPO pairs without more OpenAI calls:
# chosen = best_article
# rejected = lowest-total candidate from candidates

#optional DPO training:
python scripts/build_dpo_pairs_from_judged.py \
  --input data/train_llmsim_s2.jsonl \
  --out data/dpo_pairs_judged_kg.jsonl \
  --use_qas

# then traing with DPO pairs:
python scripts/train_dpo_news.py \
  --pairs data/dpo_pairs_judged_kg.jsonl \
  --base_model Qwen/Qwen3-4B-Instruct \
  --sft_adapter outputs/sft_all \
  --out outputs/dpo_judged_kg \
  --beta 0.1

# evaluate baseline SFT
PYTHONPATH=scripts python scripts/eval_model_llmsim.py \
  --data data/test_llmsim_s2.jsonl \
  --base_model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter outputs/sft_all \
  --out results/eval_sft_all_s5.jsonl \
  --n_simulated_readers 5 \
  --max_examples 300 \
  --use_qas \
  --pre_cache cache/pre_cache_eval.jsonl

# evaluate KG-filtered SFT
PYTHONPATH=scripts python scripts/eval_model_llmsim.py \
  --data data/test_llmsim_s2.jsonl \
  --base_model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter outputs/sft_kg_filtered_top75 \
  --out results/eval_sft_kg_filtered_top75_s5.jsonl \
  --n_simulated_readers 5 \
  --max_examples 300 \
  --use_qas \
  --pre_cache cache/pre_cache_eval.jsonl

# evaluate DPO
PYTHONPATH=scripts python scripts/eval_model_llmsim.py \
  --data data/test_llmsim_s2.jsonl \
  --base_model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter outputs/dpo_judged_kg \
  --out results/eval_dpo_judged_kg_s5.jsonl \
  --n_simulated_readers 5 \
  --max_examples 300 \
  --use_qas \
  --pre_cache cache/pre_cache_eval.jsonl

# Summarize evaluations
python scripts/summarize_eval.py \
  --files \
  results/eval_sft_all_s5.jsonl \
  results/eval_sft_kg_filtered_top75_s5.jsonl \
  results/eval_dpo_judged_kg_s5.jsonl
# Short-PHD: Detecting Short LLM-generated Text with Topological Data Analysis After Off-topic Content Insertion
Source code for the paper "Short-PHD: Detecting Short LLM-generated Text with Topological Data Analysis After Off-topic Content Insertion"

```linux
python short_phd.py \
    --data_dir data/generated_dataset \
    --filename human_text.jsonl \
    --base_lm_name meta-llama/Meta-Llama-3-8B-Instruct \
    --pred_embedding_dim 100 \
    --truncation_length 50 \
```


torchrun --standalone --nnodes=1 --nproc_per_node=8 trainer/slidewindow_trainer.py \
    --config_path config/DRT/config_63.json \
    --vocab_dir config/gpt2-small \
    --lr 2e-3 \
    --min_lr 4e-4 \
    --corpus_path ../../../antnlp/aaron.hx/corpus/pg19_gpt2/train \
    --valid_corpus_path ../../../antnlp/aaron.hx/corpus/pg19_gpt2/valid \
    --output_dir ../../../antnlp/aaron.hx/DRT_res_x63_pg19_12_60k_triton/ \
    --batch_size 1 \
    --max_seq_len 126 \
    --total_steps 60000 \
    --warm_up 0.02 \
    --accumulation_steps 1 \
    --model_type DRT \
    --log_step 50 \
    --eval_steps 5000 \
    --save_steps 5000

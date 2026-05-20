if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=60
model_name=PatchTST

root_path_name=./dataset/
data_path_name=icici_3-years.csv
model_id_name=icici
data_name=icici

random_seed=2021
pred_lens=(5 10 20 30)
total_runs=${#pred_lens[@]}
run_num=0

for pred_len in "${pred_lens[@]}"
do
    run_num=$((run_num + 1))
    echo "" >&2
    echo "============================================" >&2
    echo "  Run $run_num / $total_runs  |  pred_len = $pred_len" >&2
    echo "============================================" >&2
    python3 -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features MS \
      --target 'Close Price' \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 12 \
      --e_layers 2 \
      --n_heads 4 \
      --d_model 64 \
      --d_ff 128 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 8\
      --stride 4\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --itr 50 --batch_size 64 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
    echo "  Done  ->  logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log" >&2
done
echo "" >&2
echo "All $total_runs runs complete." >&2

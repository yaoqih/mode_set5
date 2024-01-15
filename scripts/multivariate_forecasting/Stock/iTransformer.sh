# export CUDA_VISIBLE_DEVICES=2

model_name=iTransformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/stock/ \
  --data_path SH600601.csv \
  --model_id stock_96_96 \
  --model $model_name \
  --data stock \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \h
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/stock/ \
  --data_path SH600601.csv \
  --model_id stock_96_192 \
  --model $model_name \
  --data stock \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/stock/ \
  --data_path SH600601.csv \
  --model_id stock_96_336 \
  --model $model_name \
  --data stock \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/stock/ \
  --data_path SH600601.csv \
  --model_id stock_96_720 \
  --model $model_name \
  --data stock \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.001\
  --itr 1
python predict_twoinone.py \
  --predict_data_list data/test_list.txt \
  --data_root data \
  --model_name twoinone \
  --model_path output/twoinone/best_model.pkl \
  --output_dir predict/twoinone \
  --batch_size 16 \
  --feat_dim 384 \
  --num_joints 22
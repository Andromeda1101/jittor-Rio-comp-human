export JT_STNC=1
export trace_py_var=3

python train_unified.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt --data_root data --output_dir output_unified
python predict_unified.py --predict_data_list data/test_list.txt --data_root data --pretrained_model output_unified/best_model.pkl --predict_output_dir predict_unified --batch_size 16
# python train_skeleton.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt --data_root data --model_name pct2 --output_dir output/skeleton
# python predict_skeleton.py --predict_data_list data/test_list.txt --data_root data --model_name pct2 --pretrained_model output/skeleton/best_model.pkl --predict_output_dir predict --batch_size 16
python train_skin.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt --data_root data --model_name enhanced --output_dir output/skin
python predict_skin.py --predict_data_list data/test_list.txt --data_root data --model_name enhanced --pretrained_model output/skin/best_model.pkl --predict_output_dir predict --batch_size 16


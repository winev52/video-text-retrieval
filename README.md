# video-text-retrieval
Semester Project [Fall2018]


## Usage

**training**: source/video_text_retrieval/train_vtt.py  
Example:
```shell
python train_vtt.py --mode train --model obj --resnet_path resnet_median --cap_train_path cpv20_jmet_glove/train.npy --cap_val_path cpv20_jmet_glove/val.npy --cap_test_path cpv20_jmet_glove/test.npy --num_epochs 24 --learning_rate 1e-1 --dropout 0 --log_path lr/lr0.1
```

**evaluating**: source/video_text_retrieval/train_vtt.py
```
python train_vtt.py --model both --mode test --model_path1 /objcpv20m1.0d0.5wd100es2048lr0.0001/model_best.pth.tar --model_path2 /actcpv20m1.0d0.1wd100es2048lr0.0001/model_best.pth.tar --model_path3 /flow-dtvl1cpv20m1.0d0.25wd100es1792lr0.0001/model_best.pth.tar --cap_test_path cpv20_jmet_glove/test.npy --resnet_path resnet_median
```

for more information, run the program without argument

**grid search**: source/video_text_retrieval/grid_search.py
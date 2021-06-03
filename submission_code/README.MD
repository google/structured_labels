# Causally-motivated shortcut removal using auxiliary labels

To run the waterbirds experiment:
1. Download the birds data, and the places datasets in some directory, e.g., /data/
2. To run the weighted mmd-penalized model, with MMD penalty (alpha) = 1000.0 using a kernel badwidth (sigma) = 1.0 :

```bash
python -m waterbirds.main --alpha 1000.0 --architecture pretrained_resnet --batch_size 64 --clean_back False --dropout_rate 0.0 --l2_penalty 0.0 --minimize_logits False --pflip0 0.01 --pflip1 0.01 --pixel 128 --py1_y0 0.9 --random_seed 0 --sigma 1.0 --weighted_mmd True --exp_dir /my_main_dir/my_model_dir --main_dir /data/
```





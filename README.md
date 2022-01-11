
## Momentum Contrast

The pre-training stage:

- For MoCo:
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_moco_ins.py --batch_size 128 --num_workers 24 --nce_k 16384 --softmax --moco
    ```
  
The linear evaluation stage:
- For both InsDis and MoCo (lr=10 is better than 30 on this subset, for full imagenet please switch to 30):
    ```
    CUDA_VISIBLE_DEVICES=0 python eval_moco_ins.py --model resnet50 \
     --model_path /path/to/model --num_workers 24 --learning_rate 10
    ```
  
The comparison of `CMC` (using YCbCr), `MoCo` and `InsDIS` on my ImageNet100 subset, is tabulated as below:

|          |Arch | #Params(M) | Loss  | #Negative  | Accuracy |
|----------|:----:|:---:|:---:|:---:|:---:|
|  InsDis | ResNet50 | 24  | NCE  | 16384  |  --  |
|  InsDis | ResNet50 | 24  | Softmax-CE  | 16384  |  69.1  |
|  MoCo | ResNet50 | 24  | NCE  | 16384  |  --  |
|  MoCo | ResNet50 | 24  | Softmax-CE  | 16384  |  73.4  |
|  CMC | 2xResNet50half | 12  | NCE  | 4096  |  --  |
|  CMC | 2xResNet50half | 12  | Softmax-CE  | 4096  |  75.8  |


## Acknowledgements

This code is revised by HobbitLong's project [CMC](https://github.com/HobbitLong/CMC).

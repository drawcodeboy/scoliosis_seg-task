* <a href="https://github.com/bubbliiiing/segformer-pytorch">SegFormer Codebase</a>
* <a href="https://github.com/Mr-TalhaIlyas/SegNext">SegNeXt Codebase</a>

## When Distributed Learning (2 GPUS)
### SegFormer
* 0, 1, 2 = batch-size 16
* 3, 4 = batch-size 8
* 5 = batch-size 4

### SegNeXt
* T, S, B = batch-size 16
* L = batch-size 8

## Scoliosis Results
* SegFormer-B0
    * Params: 3.7M
    * GFLOPS: 13.1552G
    * It was trained 72 EPOCHS
    * Evaluate: 100.00%
    * IoU: 0.8782 | Dice: 0.9346 | Precision: 0.9360 | Recall: 0.9357 | loss: 0.067758
    * Test Time: 00m 02s
* SegFormer-B1
    * Params: 13.7M
    * GFLOPS: 25.7578G
    * It was trained 87 EPOCHS
    * Evaluate: 100.00%
    * IoU: 0.8916 | Dice: 0.9422 | Precision: 0.9413 | Recall: 0.9453 | loss: 0.059165
    * Test Time: 00m 03s
* SegFormer-B2
    * Params: 27.3M
    * GFLOPS: 98.3070G
    * It was trained 51 EPOCHS
    * Evaluate: 100.00%
    * IoU: 0.8960 | Dice: 0.9447 | Precision: 0.9406 | Recall: 0.9509 | loss: 0.056033
    * Test Time: 00m 05s
* SegFormer-B3
    * Params: 47.2M
    * GFLOPS: 126.2161G
    * It was trained 79 EPOCHS
    * Evaluate: 100.00%
    * IoU: 0.9072 | Dice: 0.9509 | Precision: 0.9486 | Recall: 0.9552 | loss: 0.049237
    * Test Time: 00m 06s
* SegFormer-B4
    * Params: 64.0M
    * GFLOPS: 154.6397G
    * It was trained 82 EPOCHS
    * Evaluate: 100.00%
    * IoU: 0.9087 | Dice: 0.9517 | Precision: 0.9516 | Recall: 0.9538 | loss: 0.048405
    * Test Time: 00m 07s
* SegFormer-B5
    * Params: 84.6M
    * GFLOPS: 181.1287G
    * It was trained 58 EPOCHS
    * Evaluate: 100.00%
    * IoU: 0.9078 | Dice: 0.9512 | Precision: 0.9495 | Recall: 0.9548 | loss: 0.048933
    * Test Time: 00m 08s
* SegNeXt-T
    * Params: 4.8M
    * GFLOPS: 16.5449G
    * It was trained 24 EPOCHS
    * Evaluate: 100.00%
    * IoU: 0.8679 | Dice: 0.9288 | Precision: 0.9229 | Recall: 0.9372 | loss: 0.071246
    * Test Time: 00m 02s
* SegNeXt-S
    * Params: 14.6M
    * GFLOPS: 31.7697G
    * It was trained 12 EPOCHS
    * Evaluate: 100.00%
    * IoU: 0.8710 | Dice: 0.9306 | Precision: 0.9285 | Recall: 0.9351 | loss: 0.069426
    * Test Time: 00m 03s
* SegNeXt-B
    * Params: 27.6M
    * GFLOPS: 53.2278G
    * It was trained 96 EPOCHS
    * Evaluate: 100.00%
    * IoU: 0.8830 | Dice: 0.9374 | Precision: 0.9364 | Recall: 0.9404 | loss: 0.062625
    * Test Time: 00m 05s
* SegNeXt-L
    * Params: 46.2M
    * GFLOPS: 87.3279G
    * It was trained 93 EPOCHS
    * Evaluate: 100.00%
    * IoU: 0.9005 | Dice: 0.9472 | Precision: 0.9460 | Recall: 0.9505 | loss: 0.052768
    * Test Time: 00m 07s
## ICH_only Results
### threshold = 0.5
* SegFormer-B0
    * Params: 3.7M
    * GFLOPS: 13.1552G
    * It was trained 35 EPOCHS
    * IoU: 0.0848 | Dice: 0.1427 | Precision: 0.0977 | Recall: 0.4498 | loss: 0.887513
* SegNeXt-T
    * Params: 4.8M
    * GFLOPS: 16.5449G
    * It was trained 53 EPOCHS
    * IoU: 0.3833 | Dice: 0.4889 | Precision: 0.6041 | Recall: 0.4822 | loss: 0.519992
* SegNeXt-S
    * Params: 14.6M
    * GFLOPS: 31.7697G
    * It was trained 30 EPOCHS
    * IoU: 0.3950 | Dice: 0.5020 | Precision: 0.7385 | Recall: 0.4474 | loss: 0.551906
* SegNeXt-B
    * Params: 27.6M
    * GFLOPS: 53.2278G
    * It was trained 22 EPOCHS
    * IoU: 0.3648 | Dice: 0.4579 | Precision: 0.7270 | Recall: 0.4020 | loss: 0.591055
* UNet-Base
    * Params: 31.0M
    * GFLOPS: 341.0395G
    * It was trained 51 EPOCHS
    * IoU: 0.3787 | Dice: 0.4816 | Precision: 0.4092 | Recall: 0.8859 | loss: 0.804174
* SegFormer-B0 (Data Augmentation)
    * It was trained 467 EPOCHS
    * IoU: 0.0250 | Dice: 0.0473 | Precision: 0.0250 | Recall: 0.9317 | loss: 0.957689
### threshold = 0.6
* SegFormer-B0
    * It was trained 35 EPOCHS
    * IoU: 0.0850 | Dice: 0.1429 | Precision: 0.0994 | Recall: 0.4282 | loss: 0.887513
* SegNeXt-T
    * It was trained 53 EPOCHS
    * IoU: 0.3830 | Dice: 0.4883 | Precision: 0.6068 | Recall: 0.4798 | loss: 0.519992
* SegNeXt-S
    * It was trained 30 EPOCHS
    * IoU: 0.3945 | Dice: 0.5014 | Precision: 0.7435 | Recall: 0.4465 | loss: 0.551906
* SegNeXt-B
    * It was trained 22 EPOCHS
    * IoU: 0.3643 | Dice: 0.4573 | Precision: 0.7277 | Recall: 0.4011 | loss: 0.591055
* UNet-Base
    * It was trained 51 EPOCHS
    * IoU: 0.3916 | Dice: 0.4975 | Precision: 0.4250 | Recall: 0.8769 | loss: 0.804174
### threshold = 0.7
* SegFormer-B0
    * It was trained 35 EPOCHS
    * IoU: 0.0855 | Dice: 0.1433 | Precision: 0.1015 | Recall: 0.4073 | loss: 0.887513
* SegNeXt-T
    * It was trained 53 EPOCHS
    * IoU: 0.3828 | Dice: 0.4878 | Precision: 0.6097 | Recall: 0.4773 | loss: 0.519992
* SegNeXt-S
    * It was trained 30 EPOCHS
    * IoU: 0.3944 | Dice: 0.5011 | Precision: 0.7450 | Recall: 0.4456 | loss: 0.551906
* SegNeXt-B
    * It was trained 22 EPOCHS
    * IoU: 0.3637 | Dice: 0.4566 | Precision: 0.7279 | Recall: 0.4002 | loss: 0.591055
* UNet-Base
    * It was trained 51 EPOCHS
    * IoU: 0.4009 | Dice: 0.5085 | Precision: 0.4374 | Recall: 0.8683 | loss: 0.804174
### threshold = 0.8
* SegFormer-B0
    * It was trained 35 EPOCHS
    * IoU: 0.0858 | Dice: 0.1435 | Precision: 0.1040 | Recall: 0.3829 | loss: 0.887513
* SegNeXt-T
    * It was trained 53 EPOCHS
    * IoU: 0.3821 | Dice: 0.4867 | Precision: 0.6134 | Recall: 0.4742 | loss: 0.519992
* SegNeXt-S
    * It was trained 30 EPOCHS
    * IoU: 0.3934 | Dice: 0.5001 | Precision: 0.7455 | Recall: 0.4441 | loss: 0.551906
* SegNeXt-B
    * It was trained 22 EPOCHS
    * IoU: 0.3632 | Dice: 0.4559 | Precision: 0.7287 | Recall: 0.3991 | loss: 0.591055
Test Time: 00m 03s
* UNet-Base
    * It was trained 51 EPOCHS
    * IoU: 0.4098 | Dice: 0.5188 | Precision: 0.4503 | Recall: 0.8577 | loss: 0.804174
### threshold = 0.9
* SegFormer-B0
    * It was trained 35 EPOCHS
    * IoU: 0.0865 | Dice: 0.1440 | Precision: 0.1085 | Recall: 0.3500 | loss: 0.887513
* SegNeXt-T
    * It was trained 53 EPOCHS
    * IoU: 0.3815 | Dice: 0.4856 | Precision: 0.6180 | Recall: 0.4696 | loss: 0.519992
* SegNeXt-S
    * It was trained 30 EPOCHS
    * IoU: 0.3926 | Dice: 0.4991 | Precision: 0.7469 | Recall: 0.4424 | loss: 0.551906
* SegNeXt-B
    * It was trained 22 EPOCHS
    * IoU: 0.3620 | Dice: 0.4544 | Precision: 0.7290 | Recall: 0.3971 | loss: 0.591055
Test Time: 00m 03s
* UNet-Base
    * It was trained 51 EPOCHS
    * IoU: 0.4210 | Dice: 0.5315 | Precision: 0.4678 | Recall: 0.8406 | loss: 0.804174
## ICH_all Results
* 해당 방법론은 어려운 이유: 학습이 끝나면 SegFormer는 괜찮으나 SegNeXt는 모두 Negative라 판단하는 경향이 있다.
* SegFormer-B0
    * It was trained 51 EPOCHS
    * IoU: 0.3809 | Dice: 0.3911 | Precision: 0.4172 | Recall: 0.9365 | loss: 0.951504
* SegNeXt-T (??)
    * It was trained 1 EPOCHS
    * IoU: 0.8880 | Dice: 0.8880 | Precision: 1.0000 | Recall: 0.8880 | loss: 0.111111
Test Time: 00m 09s
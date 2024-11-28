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
* SegFormer-B0
    * Params: 3.7M
    * GFLOPS: 13.1552G
    * It was trained 94 EPOCHS
    * IoU: 0.2318 | Dice: 0.3146 | Precision: 0.5834 | Recall: 0.2883 | loss: 0.714832
* SegNeXt-T
    * Params: 4.8M
    * GFLOPS: 16.5449G
    * It was trained 53 EPOCHS
    * IoU: 0.3829 | Dice: 0.4900 | Precision: 0.6060 | Recall: 0.4889 | loss: 0.522744
* UNet-Base
    * Params: 31.0M
    * GFLOPS: 341.0395G
    * It was trained 51 EPOCHS
    * IoU: 0.3787 | Dice: 0.4816 | Precision: 0.4092 | Recall: 0.8859 | loss: 0.804174
* SegFormer-B0 (Data Augmentation)
    * It was trained 467 EPOCHS
    * IoU: 0.0250 | Dice: 0.0473 | Precision: 0.0250 | Recall: 0.9317 | loss: 0.957689
## ICH_all Results
* 해당 방법론은 어려운 이유: 학습이 끝나면 SegFormer는 괜찮으나 SegNeXt는 모두 Negative라 판단하는 경향이 있다.
* SegFormer-B0
    * It was trained 51 EPOCHS
    * IoU: 0.3809 | Dice: 0.3911 | Precision: 0.4172 | Recall: 0.9365 | loss: 0.951504
* SegNeXt-T (??)
    * It was trained 1 EPOCHS
    * IoU: 0.8880 | Dice: 0.8880 | Precision: 1.0000 | Recall: 0.8880 | loss: 0.111111
Test Time: 00m 09s
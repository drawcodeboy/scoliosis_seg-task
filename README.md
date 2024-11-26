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
SegFormer-B0
It was trained 94 EPOCHS
Evaluate: 100.00%
IoU: 0.2318 | Dice: 0.3146 | Precision: 0.5834 | Recall: 0.2883 | loss: 0.714832
Test Time: 00m 03s
SegNeXt-T
It was trained 53 EPOCHS
Evaluate: 100.00%
IoU: 0.3776 | Dice: 0.4830 | Precision: 0.5949 | Recall: 0.4852 | loss: 0.527711
Test Time: 00m 03s
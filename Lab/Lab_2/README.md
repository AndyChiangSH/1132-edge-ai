# **Lab 2: Quantization**

<span style="color:Red;">**Due Date: 4/24 23:59**</span>

:::success
When running `run_deit.py` and `run_slm.py`, use following command to run if you are running on the server we provide:
```bash
CUDA_VISIBLE_DEVICES=< 0 or 1 > python3 run_x.py
```
:::

## Introduction

This lab aims to learn the linear quantization, then quantize the DeiT model & Small Language Model, resulting in reduced model size and potentially improved inference speed without significant loss in accuracy.


## Part 1 : Linear Quatization Implementation (30%)

In this part, you will learn how linear quantization works and implement the quantized version of **Fully Connected Layer** (Linear) and **Convolution Layer** (Conv2d).

* Please download the provided Jupyter Notebook file using the link below. Follow the prompts and hints provided within the notebook to fill in the empty blocks.

    > [Lab2.ipynb](https://drive.google.com/file/d/18-5bdSM5JMEY9fauurWoZmMyu_F_v6BQ/view?usp=sharing)

* The MobileNetV2 will be used in this part. You can download it here:
    > [mobilenetv2_0.963.pth](https://drive.google.com/file/d/1ls20ezKCJ38spHmFflN01NyvYjNLuDoH/view?usp=sharing)

## Part 2 : Quantize Vision Transformer & Small Language Model (70%)

In this part, we will be using [HQQ](https://github.com/mobiusml/hqq) library to perform quantization on two models: `DeiT-S` and `Llama3.2-1B-Instruct`. The TAs will provide utility functions and quantization pipelines for each section to facilitate interaction with HQQ library.

* Utility Functions
    > [hqq_utils.py](https://drive.google.com/file/d/1Gst3Sdwmoh5lF2-O-UNnnAqBEB9gQfH3/view?usp=sharing)
     > [utils.py](https://drive.google.com/file/d/1NqUeXHz3toLp50F-TO4Ut6c5TpJFHGGV/view?usp=sharing) 
     > [quant_cfg.py](https://drive.google.com/file/d/1ZW8dqHsQMYazqMW6iy_nwzxJsRvSojw_/view?usp=sharing)

<!--      > [hqq_utils.py](https://drive.google.com/file/d/1e0uuus5bIzVgxby__baOtcQpz-Kzu3fL/view?usp=sharing) -->

* Environment Setting
    > [Env](https://hackmd.io/@ccyangus/Hkux8-5syg)

### Section 2.1 : Quantize DeiT-S (20%+)

Here we are performing classification task on **CIFAR100**, try to quantize the model to reduce its size as much as possible while maintaining high accuracy.

* Below is a `DeiT-S` model with 90.99% accuracy on **CIFAR100**, finetuned by TAs. You will be using this model as a starting point for quantization:

    > [0.9099_deit3_small_patch16_224.pth](https://drive.google.com/file/d/1hLFiyLRBmlcvOnm8PSRRTN69GgPiV8YT/view?usp=sharing)

* Pipeline of quantizing `DeiT-S` & performing image classification task. Use it as your reference for building your own quantization pipeline.

    > [run_deit.py](https://drive.google.com/file/d/1RP4ba1r3U9aiWCrAVTPFHO2YKUhjIpY9/view?usp=sharing)

### Section 2.2 : Quantize Llama3.2-1B-Instruct (10%)

Language models place high demand on memory, especially as the number of parameters in new models continues to grow. Therefore, quantization is one of the solutions that reduces the memory usage. However, the quality of the response may decline due to lower precision of the model parameters. 

Now, we will be performing quantization on `Llama3.2-1B-Instruct`, which is comparitively a small language model. We will use a metric called **PPL (perplexity)** to evaluate your quantized language model on **Wiki-Text**.

* Pipeline of quantizing `Llama3.2-1B-Instruct` & chatting with the SLM. Use it as your reference for building your own quantization pipeline.

    > [run_slm.py](https://drive.google.com/file/d/1xNt_TGde6dKq4UReDIG7EcH6KTvXalN-/view?usp=sharing)

### Section 2.3 : Analysis (40%)

Please hand in report in **HackMD** to answer the following questions:

1. Try to quantize `DeiT-S` from FP32 to nbit integer (n=8,4,3,2), fill in the following chart. **(group_size=64)** **(10%)**
    
|       nbit       |   32   |  8  |  4  |  3  |  2  |
|:----------------:|:------:|:---:|:---:|:---:|:---:|
|   Accuracy (%)   | 90.99  |90.98|89.59|84.76|4.29 |
| Model Size (MiB) | 82.540 |24.321|14.196|12.177|9.133|
|      Score (X)   | -45.539 |12.678|18.703|-27.577|-829.233|


2. Try to quantize `Llama3.2-1B-Instruct` from FP16 to nbit integer (n=8,4,3,2), fill in the following chart. **(group_size=64)** **(10%)**

|        nbit         |    16    |  8  |  4  |  2  |
|:-------------------:|:--------:|:---:|:---:|:---:|
|  Perplexity (PPL)   |  13.160  |     |     |     |
|  Model Size (MiB)   | 2858.129 |     |     |     |
| Throughput (toks/s) | 124.394  |     |     |     |
|      Score (X)      |    5     |     |     |     |


3. Explain how you determine the quantization method for `DeiT-S` and `Llama3.2-1B-Instruct` for best performance. If you can provide a visualized analysis or any chart according to your experiment would be better. **(15%)**

4. Which model is harder to quantize, what might be the reason ?  **(5%)**

5. Please attach screenshots showing the speedup and PPL of `Llama3.2-1B-Instruct` in your report. The screenshot will be used as the evidence in case performance drops due to different hardware platform. **(For Criteria of Section 2.2)**

## Hand-In Policy

You will need to hand-in:
* Fill out Part 1 in ***Lab2.ipynb***, and rename it to ***`<YourID>_part1`.ipynb***

* Implement your own qunatization configuration in ***`quant_cfg.py`*** 
    
* ***`url.txt`*** should include the URL of your HackMD report.

Please organize your submission files into a zip archive structured as follows:

```scss
YourID.zip     
    ├── YourID_part1.ipynb
    ├── quant_cfg.py
    └── url.txt
```

:::warning
There should be **No Folder** in the zip file
Also, make sure the TAs have permission to access your HackMD report
:::

:::info
Any modification in `run_deit.py` & `run_slm.py` are **not allowd**, but you can add additional functions  in `quant_cfg.py` to support your `get_quant_config_x` functions.
:::
## Evaluation Criteria

1. For `DeiT-S`, TAs will run `run_deit.py` and load quantization configuration with your `get_quant_config_deit` function. The Score will be calculated base on the criteria below:

$$
  Score = 20 - Max(90 - Accruacy, \ 0) \times 10 + 17 - Model\_Size (MiB)
$$


2. For `Llama3.2-1B-Instruct`, TAs will run`run_slm.py` and load quantization configuration with your `get_quant_config_slm` function. The Score will be calculated base on the criteria below: 

    ```python
    score = 0
    score += 5 if PPL <= 14 else 0
    score += 5 if speedup >= 1.3
    ```
    
## Reference
- [Introduction to Quantization - 1](https://medium.com/@anhtuan_40207/introduction-to-quantization-09a7fb81f9a4)
- [Introduction to Qunatization - 2](https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c/)
- [Visual Guide of Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
- [Quantization Granularity](https://medium.com/@curiositydeck/quantization-granularity-aec2dd7a0bb4)
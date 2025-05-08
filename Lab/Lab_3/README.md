# **Lab 3: Triton GPU programming for neural networks**

<span style="color:Red;">**Due Date: 5/8 23:55**</span>

## Introduction

Programming for accelerators such as GPUs is critical for modern AI systems. This often means programming directly in proprietary low-level languages such as CUDA. Triton is an alternative open-source language that allows you to code at a higher-level and compile to accelerators like GPU.

This lab is meant to teach you how to use Triton from first principles in an interactive fashion. You will start with trivial examples and build your way up to real algorithms like Flash Attention and Quantized neural networks. Through this hands-on experience, you will learn about basic GPU programming.

* Please download the provided Jupyter Notebook files using the link below.
Follow the prompts and hints provided within the notebook to fill in the empty blocks and answer the questions.

    > [Lab3_Part1.ipynb](https://drive.google.com/file/d/1gfqmgNv0LgaiFbshBcUAS4DJ8BOabP8k/view?usp=drive_link)
    > [Lab3_Part2.ipynb](https://drive.google.com/file/d/1ZOnQE4e6_SHfRqJNnQUp9644SGW6Q77s/view?usp=drive_link)
    
## Environments
Before doing this lab, you will need to install following 4 libraries to run the code.

* **PyTorch & Triton**
```
pip install torch==2.6.0
pip install triton==3.2.0
```

* **Triton-Viz**
```
git clone https://github.com/Deep-Learning-Profiling-Tools/triton-viz.git
cd triton-viz
pip install -e .
```

* **jaxtyping**
```
pip install jaxtyping
```

* **Pycairo**
```
sudo apt update
sudo apt install -y libcairo2-dev pkg-config
pip install pycairo
```

* **Matplotlib**
```
pip install matplotlib
```

## Part 1: Trivial examples (60%)
In this part, you will start with trivial examples. 

You will specifically learn about:

* The basic programming model of Triton.
* Pointer arithmetic.



## Part  2: Matrix Multiplication in Triton  (40%)

In this part, you will write a very short high-performance FP16 matrix multiplication kernel.

You will specifically learn about:

* Block-level matrix multiplications.
* Multi-dimensional pointer arithmetic.
* Program re-ordering for improved L2 cache hit rate.




## Grading

* 1. Constant Add Block - 5%
* 2. Outer Vector Add - 5%
* 3. Outer Vector Add Block - 5%
* 4. Fused Outer Multiplication - 5%
* 5. Long sum - 5%
* 6. Long softmax - 7%
* 7. Simple Flashattention - 13%
* 8. Quantized Matrix Mult - 15%
* 9. Matrix Mult - 20%
* 10.  Faster Matrix Mult - 20%


## Hand-In Policy

You will need to hand-in:
* Fill out both ***Lab3_Part1.ipynb*** and ***Lab3_Part2.ipynb***
* Remember to keep the ***outputs*** of the Jupyter Notebook files, it will be easy for TAs to check.
* Rename them to ***```<YourID>```_Part1.ipynb*** and ***```<YourID>```_Part2.ipynb*** respectively.
* ***```<YourID>```.zip***
  - ***```<YourID>```_Part1.ipynb***
  - ***```<YourID>```_Part2.ipynb***



## Penalty
* Wrong Format - 5%
* Late Submission - 10% per day



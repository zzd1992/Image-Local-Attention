# Image Local Attention: a Better PyTorch Implementation

## Introduction

Attention is widely used in deep learning now. Given a query, a collection of keys and values. The output of an attention module is the weighted sum of all values. The weights are obtained based on the similarities between the query and keys which are usually measured by their inner products. However, when number of keys is large, it is cost to apply such a module.

Researchers consider about local attention to address this problem which a small subset of keys are involved given a query. For images, "local" means a image patch around a pixel. Image local attention achieves great success on image restoration tasks. However, current implementations are based on the `im2col` opreation which is memory expensive espesially when the local patch is large.

## Implementation

Here, queries `Q`, keys `K` and value `V` are represented in `CHW` (channel, height, width) tensors.

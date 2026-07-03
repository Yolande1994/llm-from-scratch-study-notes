# 更高效的多头注意力实现

- [mha-implementations.ipynb](mha-implementations.ipynb) 包含并比较了不同的多头注意力实现方式

### 总结

下图总结了性能基准测试（越低越好）。

&nbsp;
#### 仅前向传播

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/1_forward-only.webp?1" width="500px"></a>

&nbsp;
#### 前向和反向传播

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/2_forward-and-backward.webp?1" width="500px"></a>

&nbsp;
#### 编译后的前向和反向传播

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/3_forward-and-backward-compiled.webp?1" width="500px"></a>
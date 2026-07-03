# **预训练的超参数优化（Optimizing Hyperparameters for Pretraining）**  

[`hparam_search.py`](hparam_search.py) 脚本基于 **[附录 D: 训练循环优化](../../appendix-D/01_main-chapter-code/appendix-D.ipynb)** 中扩展的训练函数，  
用于 **通过网格搜索（Grid Search）寻找最佳超参数配置**。  

> **💡 注意**  
> 该脚本运行时间较长，建议在 `HPARAM_GRID` 字典中减少搜索的超参数组合，以加快实验速度。  
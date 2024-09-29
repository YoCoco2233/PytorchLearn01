# Vector @YoCoco2233 9/23/2024
import torch

x = torch.arange(4)
print(x)
print(x[3])
print(len(x))
print(x.shape)
"""
请注意，维度（dimension）这个词在不同上下文时往往会有不同的含义，这经常会使人感到困惑。 
为了清楚起见，我们在此明确一下： 向量或轴的维度被用来表示向量或轴的长度，即向量或轴的元素数量。 
然而，张量的维度用来表示张量具有的轴数。 在这个意义上，张量的某个轴的维数就是这个轴的长度。
"""
# 论文阅读知识库
优秀论文

学习进度：
 2025.8.1：通过对ai的总结和知识的提问，对于《GALLa Graph Aligned Large Language Models for Improved Source Code Understanding》的内容已经大致理解清楚，并优化了笔记。剩余看看原文和代码工程的实现。
2025.8.11：《Trainable Dynamic Mask Sparse Attention.md》这篇论文还是比较有意思的，主要还是一种新的可训练动态掩码稀疏注意力，主要还是解决自注意力对于每个Token都去做一个注意力关系，这个是太耗费资源的，个人知识储备中的如informer论文中实验证明了其实每个Token之间并不是都是有关系的，只有部分有关系，而我们全都去做一个注意力的话，这样太耗费资源了，没必要。因此本文说的这种注意力对于在处理文本对话中是比较有作用的，因为对话随着轮数增加，token一直在增加，越到后面负担越重，使用这个稀疏注意力可以大大减少token。

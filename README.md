### README

--------------

_language:中文_



#### Environment

---------

1. sklearn
2. python3.6
3. Ubuntu18.04



#### The Refence Paper

-------------

#### DOI: 10.1016/j.protcy.2013.12.451

1. 不同之处在于: 文章里用的L-M 算法作为训练算法，我采用的是sklearn中现成的算法,在加大样本量后(从700多到6000多)，epoch(max_iter)=2500,训练算法采用"adam",得分率去到0.95067...

   <img src="./loss_function.png" alt="loss_function" style="zoom:50%;" />

2. 维持低样本量时,max_iter=2500,采用"lbfgs",得分率可去到0.96....



#### 心得

------------------------

1. 样本量低时请使用“lbfgs”;
2. 训练数据一定要进行正则化，使其变化到[-1,1]之间;
3. 受限与sklearn的nn网络设置的影响，只能对adam画出损失函数的图像。后续可考虑采用TensorFlow
4. 损失函数中部分概念还不清楚。需要进一步学习。
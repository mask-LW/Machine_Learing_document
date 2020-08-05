# Dropout详解

## Dropout Regularization

训练阶段：

![image-20200730182333306](https://i.loli.net/2020/07/30/pWIOPzuTfsjQy39.png)

训练上图这样的神经网络，它存在过拟合，这就是**dropout**所要处理的.

**Each time before updating the parameters**

每次更新参数之前：

每个神经元有p%发生dropout，导致实际训练时所使用神经网络的结构发生变化。

![image-20200730182720800](https://i.loli.net/2020/07/30/i7p6PFyEgjcbMm4.png)

每个mini-batch都会重新取样神经元。



测试阶段：神经元不再采用dropout，而是所有参数使用。

若 p% = 50%，则训练过程得到的所有参数乘上(1-p)%。

![image-20200730182953872](https://i.loli.net/2020/07/30/lJt7yRD92fVITAE.png)



**Dropout**实现：

最常用的实现方法：**inverted dropout**（反向随机失活）



首先定义一个向量d，用一个三层（l=3）网络来举例说明

d3 = np.random.rand(a3.shape[0],a3.shape[1])

然后看它是否小于某数，我们称之为**keep-prob**，**keep-prob**是一个具体数字，它表示保留某个隐藏单元的概率，此处**keep-prob**等于0.8，它意味着消除任意一个隐藏单元的概率是0.2，它的作用就是生成随机矩阵。



从第三层中获取激活函数a

a3 =np.multiply(a3,d3)，它的作用就是让d3中所有等于0的元素（输出），而各个元素等于0的概率只有20%，乘法运算最终把d3中相应元素输出，即让d3中0元素与a3中相对元素归零。



最后，我们向外扩展a3，用它除以0.8，或者除以**keep-prob**参数。

a3 = a3/keep-prob



如果**keep-prop**设置为1，那么就不存在**dropout**，因为它会保留所有节点。反向随机失活（**inverted dropout**）方法通过除以**keep-prob**，确保a3的期望值不变。



## Dropout is a kind of ensemble

训练 Ensemble modle

![image-20200730183726402](https://i.loli.net/2020/07/30/yNJZmqW4Yw7gcEK.png)



![image-20200730183851075](https://i.loli.net/2020/07/30/uKqQIOiUVytmn4J.png)

使用一个mini-batch去训练模型，M个神经元有2^m种神经网络

![image-20200730184641048](/Users/mac/Library/Application Support/typora-user-images/image-20200730184641048.png)



神奇的是，此处使用dropout平均出来的结合近似于神经元不使用dropout但参数使用的结果。

![image-20200730184739197](https://i.loli.net/2020/07/30/UlJZogH7Bfn8iT2.png)


---
layout: post
math: true
title: "Chain Rule and Back Propagation"
categories: deep_learning
tags: [hands-on, back-propagation, calculus]
---

# 链式法则和反向传播

## 站在初中生的角度

深度学习就像教计算机玩一个‘猜答案’的游戏。

- 初始猜测：计算机根据简单规则（比如随机数）做出第一次猜测。

- 计算误差：它会用数学方法算出猜测和正确答案差了多少（比如‘差了5分’）。

- 调整参数：计算机自动修改内部的‘计算参数’（像大脑神经元的连接强度），让下次猜测更准。

- 反复练习：用数百万条数据训练后，它的猜测会越来越精准（比如能认出猫狗或预测天气）。

## 过程梳理

前面推导了对复合函数求导的链式法则。在此基础上，再来一遍于反向传播应用的过程：

训练开始以后，先进行前向传播，计算出输出值。

1. 前向传播：计算机根据输入数据和当前参数，计算出输出值 $z = w \cdot x + b$。
2. 计算输出(预测值)：计算机用激活函数（比如 sigmoid）处理输出值，得到预测值 $y = \sigma(z) = \frac{1}{1 + e^{-z}}$。
3. 计算误差：计算机用损失函数（比如均方误差）计算预测值和真实值之间的差异，得到误差 $L = \frac{1}{2}(y - y_{true})^2$。

开始反向传播，计算梯度：

目标是计算损失函数 $L$ 对参数 $w$ 的梯度，以便更新参数。

1. 计算 $dL/dy$：损失函数对预测值的梯度，得到 $\frac{dL}{dy} = y - y_{true}$。因为 $L= \frac{1}{2}(y - y_{true})^2$。
2. 计算 $dy/dz$：激活函数对输入的梯度，得到 $\frac{dy}{dz}= \sigma'(z) = y(1-y)$。因为 $\sigma'(z)=\sigma(z)(1 - \sigma(z)) = y(1-y)$。
3. 计算 $dz/dw$：线性层对权重 $w$ 的梯度，得到 $\frac{dz}{dw} = x$。因为 $z = w \cdot x + b$，对 $w$ 求导得到 $x$。
4. 根据链式法则，对上述进行组合，得到 $\frac{dL}{dw} = \frac{dL}{dy} \cdot \frac{dy}{dz} \cdot \frac{dz}{dw}= (y - y_{true}) \cdot y(1-y) \cdot x$。
5. 更新权重：根据计算出的梯度，使用梯度下降法更新权重 $w$，得到 $w = w - \eta \cdot \frac{dL}{dw}$，其中 $\eta$ 是学习率。

通过以上步骤反复迭代，神经网络可以逐步调整参数，使得预测值越来越接近真实值。这就是反向传播的核心思想。

## 实例计算

举一个计算的例子：

假设我们有一个简单的神经网络，输入 $x = 0.5$，权重 $w = 0.4$，偏置 $b = 0.1$，学习率 $\eta = 0.01$，真实值 $y_{true} = 0.8$。

前向传播：

1. 计算 $z = w \cdot x + b = 0.4 \cdot 0.5 + 0.1 = 0.2 + 0.1 = 0.3$。
2. 计算预测值 $y = \sigma(z) = \frac{1}{1 + e^{-0.3}} \approx 0.574$。
3. 计算误差 $L = \frac{1}{2}(y - y_{true})^2 = \frac{1}{2}(0.574 - 0.8)^2 \approx 0.025$。

接下来进行反向传播：
1. 计算 $dL/dy = y - y_{true} = 0.574 - 0.8 = -0.226$。
2. 计算 $dy/dz = y(1-y) = 0.574 \cdot (1 - 0.574) \approx 0.244$。
3. 计算 $dz/dw = x = 0.5$。
4. 根据链式法则，计算梯度 $\frac{dL}{dw} = (y - y_{true}) \cdot y(1-y) \cdot x = -0.226 \cdot 0.244 \cdot 0.5 \approx -0.027$。
5. 更新权重 $w = w - \eta \cdot \frac{dL}{dw} = 0.4 - 0.01 \cdot (-0.027) = 0.4 + 0.00027 \approx 0.40027$。

上面第4步，计算出的梯度为负，说明当前的预测值低于真实值，因此需要增加权重 $w$ 来提高预测值。

## 反向传播和大脑

深度学习借鉴了大脑的神经元连接方式。大脑中的神经元通过突触连接，传递信号并调整连接强度。类似地，神经网络中的权重就是这些连接强度。

但是，反向传播机制在大脑中并不存在。我们通常说大脑是个精密的机器，但这取于对“精密”的定义。深度学习的反向传播算法是一个数学优化过程，链式反应的过程也注定了如果过程中出现了一点错误，那么链式法则的应用会把整个错误给放大化。

通常有两种情况会导致反向传播的错误：

1. 梯度爆炸：深度学习的网络层数通常非常深，以100层为例，如果每层的梯度值超过1，那么链式法则的累积效应会使得多层连乘以后的梯度呈指数增长，导致权重更新过大，从而使模型不稳定。

2. 梯度消失：在深层网络中，随着层数的加深，梯度也可能向另一个极端转化，会变得非常小，导致权重更新缓慢，甚至停止学习。这在使用sigmoid等激活函数时尤为明显，因为它们在输入较大或较小时梯度接近于零。

而相对而言，人脑的“容错性”更高。大脑的神经元连接是通过化学信号传递的，具有一定的冗余和适应性。大脑在噪声、损伤下仍能进行学习，而在反向传播过程中，如果网络结构出现出现问题，学习将无法完成。

大脑的神经元之间没有精确的数学导数计算，突触调整依赖于化学过程（如钙离子浓度）。而反向传播的机制决定了网络结构必须是对称的前后向连接，否则参数调整无法完成。

大脑的学习过程是局部的、异步的、基于生物化学信号的，而反向传播是全局的、同步的、基于数学优化的。大脑可以通过“小样本”进行学习，而反向传播通常需要大量的数据来进行有效的训练。

大语言模型的出现，意味着在现有的数学工具的基础之上，计算机学习可以达到的巅峰，但是仍然从学习的效率上无法与人脑相比。

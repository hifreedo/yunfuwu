---
layout: post
math: true
title: "Chain Reaction and Chain Rule"
categories: deep_learning
tags: [hands-on, math, calculus]
---

# 链式反应和链式法则

之前的文章给自己挖了一个坑，深度学习的核心在于反向传播，而反向传播的核心在于求导，求导的核心在于链式法则。

当一个中子轰击一个铀235原子核的时候，会分裂出3个新的中子，3个新的中子又会轰击更多的原子核，进而又产生更多的中子。链式反应是人类迄今掌握的最强能量释放机制。

链式反应的效应类似于人类大脑的神经元传递信息的方式。一个神经元的激活会影响到下一个神经元，进而影响到更多的神经元。在百万年的进化中，人类大脑实现了非常高效的信息传递和处理机制，即使在高强度思考的情况下，能耗也仅约为22瓦特。

而一台GPU加持的普通家用电脑的能耗轻松超过500瓦特，简而言之，计算很昂贵。而链式法则是人们发明的数学工具，基于它，反向传播的计算成为可能，也让深度学习能够真正落地。

链式法则是微积分中的一个重要定理，它描述了复合函数的导数如何计算。具体来说，如果有两个函数 $f(x)$ 和 $g(x)$，那么它们的复合函数 $h(x) = f(g(x))$ 的导数可以通过链式法则来求解：

$$
h'(x) = f'(g(x)) \cdot g'(x)
$$

上面的公式表示，$h(x)$ 的导数 $h'(x)$ 等于 $f(x)$ 在 $g(x)$ 处的导数 $f'(g(x))$ 乘以 $g(x)$ 的导数 $g'(x)$。

## 链式法则的推导

下面是对链式法则的推导：

$$

h'(x) = \lim_{\Delta x \to 0} \frac{f(g(x + \Delta x)) - f(g(x))}{\Delta x}

$$

以上是基于导数的定义。但是到这一步以后好像很难再向前推进。

链式法则的推导有两个关键步骤，接下来是第一个关键步骤，在式子中乘以 $g(x + \Delta x) - g(x)$ 分之 $g(x + \Delta x) - g(x)$：

$$
= \lim_{\Delta x \to 0} \frac{f(g(x + \Delta x)) - f(g(x))}{\Delta x} \cdot \frac{g(x + \Delta x) - g(x)}{g(x + \Delta x) - g(x)}

$$

对上式进行移项变形：

$$
= \lim_{\Delta x \to 0} \frac{f(g(x + \Delta x)) - f(g(x))}{g(x + \Delta x) - g(x)}\cdot \frac{g(x + \Delta x) - g(x)}{\Delta x}
$$

可得右项就是 $g(x)$ 的导数：

$$
= \lim_{\Delta x \to 0} \frac{f(g(x + \Delta x)) - f(g(x))}{g(x + \Delta x) - g(x)}\cdot g'(x)
$$

接下来是第二个关键步骤，令 $g(x) = k$ , 则 $g(x+\Delta x) = k + \Delta k$，其中 $\Delta k = g(x+\Delta x) - g(x)$。

$$
= \lim_{\Delta x \to 0} \frac{f(k + \Delta k) - f(k)}{k + \Delta k -k} \cdot g'(x)

$$

化简：

$$
= \lim_{\Delta x \to 0} \frac{f(k + \Delta k) - f(k)}{\Delta k} \cdot g'(x)
$$

$$
= f'(k) \cdot g'(x)

$$

证毕：

$$
= f'(g(x)) \cdot g'(x)
$$

链式法则的变化形式：

基于：

$$
= \lim_{\Delta x \to 0} \frac{f(g(x + \Delta x)) - f(g(x))}{g(x + \Delta x) - g(x)}\cdot \frac{g(x + \Delta x) - g(x)}{\Delta x}
$$

令: $y = f(g(x))$，则：

$$

= \frac{dy}{dk} \cdot \frac{dk}{dx}

= \frac{dy}{dx}

$$

链式法则的链式反应可以继续延伸：

令：$y = f(u)$，$u = g(v)$, $v = h(x)$，则：

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dv} \cdot \frac{dv}{dx}
$$

## 对sigmoid函数求导

基于链式法则对sigmoid函数的导数进行推导：

$$

\sigma(x) = \frac{1}{1 + e^{-x}}

$$

令: $t = 1 + e^{-x}$，则 $\sigma(x) = \frac{1}{t}$：

$$

\sigma'(x) = -{t^{-2}}
$$

对 $t$ 求导：

$$
t' = 0 + -e^{-x}

$$

根据链式法则：

$$
\sigma'(x) = \frac{d(s)}{d(t)} \cdot \frac{d(t)}{d(x)} = -\frac{1}{t^2} \cdot (-e^{-x}) = \frac{e^{-x}}{t^2}
$$

将 $t$ 代入：
$$
\sigma'(x) = \frac{e^{-x}}{(1 + e^{-x})^2} = \sigma(x) \cdot (1 - \sigma(x))
$$

由此得到sigmoid函数的导数计算的简单方法。
保持足够的简单，才有最大的实用空间。

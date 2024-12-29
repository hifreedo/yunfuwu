---
layout: post
toc: true
title: "深度学习的最核心"
categories: deep_learning
tags: [derivative]
---

## 核心

* The core is derivative *

在MS的时候，以前经常问候选人的傻瓜测试问题是：
如何用通俗的语言，简单几句话向小学生介绍什么是数据库？

在那个年代，工程离不开关系型数据库，SQL Server是微软的三驾马车之一。
现在，LLM是新的生产力革命。

LLM背后是深度学习，深度学习的核心环节是反向传播。

反向传播里面，数学基础是求导，评估函数的斜率，以适当地评估输入的变化，在对于输出的结果当中的贡献大小。

因此，深度学习包括：神经元、网络结构定义以及反向传播函数的基本实现，一个迷你的toy，百余行代码已经足够。

其它所有的训练、推理框架和算法，只为了解决一个问题：

效率。

Be able to dance with PB data and Billions of parameters.


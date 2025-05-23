---
layout: post
title: "Coding Agent"
categories: engineering
tags: [coding, coding-agent]
---

直到当前为止，Coding Agent 从事稍有复杂度的线上工程还是不行的，无论是 Copilot, Cursor 还是windsurf。
三个agent都有使用过，如果项目管理的代码行数超过3千行（不包括三方包和库在内），无论哪个agent都是力不从心，使用起来犹如钝刀割肉。

有些产品经理拍视频称，在自己敲0代码的基础上，做出了一个软硬件结合的复杂应用。那是一个demo，上不了线。

仍然，coding agent的横空出现，仍然是一个了不起的变革，只是和一辆赛车一样，它是需要人来驾驭的工具。为了更好地发挥出它的性能，需要：

* 优秀的表达能力，对于需要开发的产品功能的表达、技术名词和术语的表达；
* 项目管理的能力，agent和人一样，如果你喂给它的功能过于庞大，它也吃不了，这个时候需要学会恰当地划分开发的粒度，先做什么、后做什么；
* 代码调试的能力，除了反馈给它你看到了什么，你需要主动建议agent在哪里输出诊断和日志，合作才能高效定位代码中的问题；
* 代码管理的能力，对于你给出的需求，agent倾向于首先考虑把它完成，而不是考虑在系统最优的前提下完成，所以agent也会堆叠代码构建屎山。对于重复的代码，你需要明确指出，不要重复发明轮子；
* 技术选型的能力，agent无法了解当下硬件限制、业务背景以及未来规模的变化，即使你有事先输入，它也不会总是从全局出发考虑。比如，关于客户端通信，需要做Polling还是Long Polling还是用Websocket? ASGI服务器使用Daphne还是Uvicorn，你可以获取信息，但决定还是需要自己来完成；
* 安全管理的能力，当下agent的编程策略是写出最小可运行的版本，如果你不是专门提出要求，它不会在代码的安全性、健壮性上面做出额外考虑；
* 自己动手的能力，现在的agent可以是当之无愧的熟练工，但是它也有自己的明显的盲区或者幻觉区域，有的时候在反反复复修正它自己写的一个bug而不能成功的时候，你需要卷起袖子亲自下场，人机结合，才能走出漩涡。
* 版本管理的能力，对于一个较大的任务，在多次迭代之后，甚至会发生前面的功能都不可用，一夜回到解放前的状况，你需要步步为营，及时为新完成的功能建立版本管理，才能小步快跑。

基于以上，对于目前的 Coding Agent 能够替代真正的程序员这一想法，基本上是个“幻觉”。

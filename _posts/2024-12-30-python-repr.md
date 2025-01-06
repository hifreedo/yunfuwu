---
layout: post
title: "repr and str"
categories: engineering
tags: [python]
---

__repr__(self)是Python中的特殊方法，作用是把对象进行文本化的表达。
一个定义良好的repr方法应该有助于重建对象，从而帮助debug。

repr 应该是 representation 表示、象征的缩写。
__repr__ 以及后面提到的 __str__ 并不是Python中的奇技淫巧，而是体现了这门语言的设计理念。

比如象这样：

```Python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __repr__(self):
        return f'Person(name={self.name!r}, age={self.age})'

# Creating an instance of Person
person = Person('Alice', 30)

# Using __repr__ to represent the person object
print(repr(person))
# Output: Person(name='Alice', age=30)
```

通过使用repr，可以打印出当前的对象，它有两个属性。
{self.name!r}中的!r指示当前的输出应该依据repr的格式输出，因此我们将会得到name='Alice'。
指明这是一个字符串，而不是直接输出name=Alice。

如果不使用repr或者str方法，直接输出person对象，我们将得到：
<__main__.Person object at 0x11aa03ed0>
它对于理解当前程序运行没有什么帮助。

因此，在日志系统中，repr特别有用：

```Python
import logging

logging.basicConfig(level=logging.DEBUG)

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return f'Person(name={self.name!r}, age={self.age})'

person = Person('Alice', 30)
logging.debug(repr(person))
```

上面的代码运行将会输出：
DEBUG:root:Person(name='Alice', age=30)

除了Debug以外，repr还用在Ipython等交互式编程场景中。

除了repr，在Django框架里，我们还会频繁地和__str__打交道。

```Python
from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)
    birth_date = models.DateField()

    def __str__(self):
        return self.name

```

上面的模型定义里，使用了__str__，它将只会返回当前的用户名字，而不是对象的完整属性。
在template页面，默认的，Author instance将会直接以name字段展示给用户。

__repr__方法主要面向开发人员，它强调对于对象的清晰而准确的输出。

__str__方法主要面用户，展示的是我们希望向用户展示的信息。

就象是厨房做饭，我们只用一把刀也可以，但是现在我们准备了两把刀子，一把专门用于切生肉，一把用于切熟肉。

这就是这门语言的设计理念。

## ChatHandler

1. send, 发送一条消息到网页端
2. recv, 从网页端接收消息

## run, 启动web服务, `run(chat, title="chat", web_port=9999, log_dir=None)`

1. chat, 具体对话逻辑, 接受两个参数, client, uid
  1. client, 在run内部可以使用send(msg), recv()和网页通信
  2. uid, 客户端识别ID
2. title, 网页显示的标题, 默认 chat
3. web\_port, 服务监听的端口, 默认 9999
4. log\_dir, 日志目录
  1. 默认是None, 不输出日志
  2. 输出日志, 对于不同的客户端, 在run函数中将日志写入uid对应文件 log_dir/uid.txt

## 依赖

1. pip install tornado

## 例子

```python
# coding: utf-8

from __future__ import print_function
from __future__ import absolute_import

import sys
sys.path.append("/path/to/chat_web")
from chat_web import run


def chat(client, uid):
    client.send("你好")
    msg = client.recv()
    client.send(u"你好")
    msg = client.recv()
    while True:
        client.send(msg[::-1])
        msg = client.recv()

run(chat)
```

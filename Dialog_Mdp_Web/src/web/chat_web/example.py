# coding: utf-8

from __future__ import print_function
from __future__ import absolute_import

import sys
import argparse

sys.path.append(".")
from chat_client import run


def chat(client, uid):
    client.send("您好，请问您是要办电业务吗～")
    msg = client.recv()
    client.send(u"你好")
    msg = client.recv()
    while True:
        client.send(msg[::-1])
        msg = client.recv()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='web')
    parser.add_argument('-web_port', type=int, default=9999)
    parser.add_argument('-title', default="Reverse")
    parser.add_argument('-log_dir')
    parser.add_argument('-local_prompt', default="➜ ")
    opt = parser.parse_args()
    print(opt)
    run(chat, opt.mode, opt.title, opt.web_port, opt.log_dir, opt.local_prompt)

# coding: utf-8

from __future__ import print_function
from __future__ import absolute_import

import sys
import argparse
import FSM_MDP
import KG

sys.path.append("src/web/chat_web")
from chat_client import run

kg = KG.KG()
tree = kg.build_tree()
fsm = FSM_MDP.FSM()

class IO:
    def __init__(self, if_web, client):
        self.if_web = if_web
        self.client = client

    def in_fun(self):
        if self.if_web:
            in_res = self.client.recv()
        else:
            in_res = input()
        return in_res

    def out_fun(self, out_res):
        if self.if_web:
            self.client.send(out_res)
        else:
            print(out_res)


def chat(client, uid):
    io_method = IO(if_web, client)
    fsm.process(io_method, intent_array, state_array, state_linklist, disambiguation_node_dict, 20)




if __name__ == '__main__':
    array_len = tree.size()+1  # 因为多了一个不在树中的unk
    state_array, intent_array = [0] * array_len, [0] * array_len
    state_linklist, disambiguation_node_dict = [], {}
    if_web = False  # if you are in debug mode, set False
    if if_web:
        parser = argparse.ArgumentParser()
        parser.add_argument('-mode', default='web')
        parser.add_argument('-web_port', type=int, default=9999)
        parser.add_argument('-title', default="Reverse")
        parser.add_argument('-log_dir')
        parser.add_argument('-local_prompt', default="➜ ")
        opt = parser.parse_args()
        print(opt)
        run(chat, opt.mode, opt.title, opt.web_port, opt.log_dir, opt.local_prompt)
    else:
        io_method = IO(if_web, None)
        fsm.process(io_method, intent_array, state_array, state_linklist, disambiguation_node_dict, 20)






# coding: utf-8

from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import time

if sys.version_info[0] == 2:
    import Queue as queue
    input = raw_input
else:
    import queue

from threading import Thread

from tornado import web, websocket, ioloop


def run(chat,
        mode='web',
        title="chat",
        web_port=9999,
        log_dir=None,
        local_prompt=">>:"):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if mode == 'web':
        class IndexHandler(web.RequestHandler):
            def get(self):
                self.render(
                    os.path.join(base_dir, 'templates', 'index.html'),
                    title=title,
                    show_log=(log_dir is not None))

        class ChatHandler(websocket.WebSocketHandler):
            def open(self, uid):
                self.uid = uid
                print("%s connected." % uid)
                self.thread = Thread(target=chat, args=(self, uid))
                self.thread.setDaemon(True)
                self.msg = queue.Queue(maxsize=2)
                self.thread.start()

            def on_message(self, msg):
                if self.msg.empty():
                    self.msg.put(msg)
                    self.msg.put(999)

            def on_close(self):
                print("%s disconnected." % self.uid)

            def send(self, msg):
                self.write_message("999 888 "+ msg)


            def recv(self):
                # recv 和 on_message 一个是从队列里取 一个是往队列里面放
                msg = self.msg.get()
                self.msg.get()

                return msg

        handlers = [
            (r'/', IndexHandler),
            (r'/ws/(.*)', ChatHandler),
            (r'/static/(.*)', web.StaticFileHandler, {
                'path': os.path.join(base_dir, 'static')
            }),
        ]

        if log_dir is not None:
            handlers.append((r'/log/(.*)', web.StaticFileHandler, {
                'path': log_dir
            }))

        app = web.Application(handlers)
        web_port = int(web_port)
        app.listen(web_port)
        ioloop.IOLoop.instance().start()

    else:

        class ChatHandler:
            def send(self, msg):
                print(msg)

            def recv(self):
                msg = input(local_prompt)
                return msg

        chat(ChatHandler(), time.strftime("%Y%m%dT%H:%M:%S", time.gmtime()))

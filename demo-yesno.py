#
# Copyright (c) 2024, Alex J. Champandard.
#

import sys
import time
import random
import numpy as np

import vispy
from vispy import app
from vispy import scene
from vispy.scene.visuals import Text

import device


ANSWERS = ["-", "Yes", "No"]


class Application(object):
    def __init__(self):
        self.gather_thread = device.GatheringThread()

        vispy.app.use_app('glfw')
        self.timer = app.Timer(0.1, connect=self.on_tick, start=True)

        self.canvas = scene.SceneCanvas(
            title="Capture",
            size=(1280, 720),
            keys="interactive",
            bgcolor="#111111",
            px_scale=2,
        )

        self.answer_text = None
        self.answer_time = 0.0
        self.score_time = 0.0
        self.score_average = 0.0
        self.ticks = 0
        self.start = None

        t1 = Text("-", parent=self.canvas.scene, color="#C0C0C0")
        t1.font_size = 24
        t1.pos = self.canvas.size[0] * 1 // 2, self.canvas.size[1] // 2
        self.answer_widget = t1

        t3 = Text("", parent=self.canvas.scene, color="#404040")
        t3.font_size = 6
        t3.pos = self.canvas.size[0] * 1 // 2, self.canvas.size[1] - 32
        self.time_widget = t3

        self.canvas.events.mouse_press.connect(self.on_click)
        self.canvas.events.close.connect(lambda evt:self.shutdown)

    def shutdown(self):
        self.gather_thread.stop()

    def on_click(self, evt):
        pass

    def update_target(self):
        if self.answer_text != '-':
            self.answer_text = '-'
            self.arrow_switch += (3.0 + random.random() * 2.0)
        else:
            self.answer_text = random.choice(ANSWERS[1:])
            self.arrow_switch += (0.5 + random.random() * 1.5)

        self.answer_widget.text =self.answer_text
        self.gather_thread.current_targets = ANSWERS.index(self.answer_text)

        self.canvas.update()

    def on_tick(self, evt):
        if self.start is None:
            self.start = time.time()
            self.arrow_switch = self.start
        else:
            self.ticks += 1

        if time.time() > self.arrow_switch:
            self.update_target()

        if self.ticks == 2:
            self.gather_thread.start()
        if not self.gather_thread._active:
            self.canvas.app.quit()

        self.time_widget.text = '%is' % int(evt.elapsed)       
        return

    def run(self):
        self.canvas.show()
        self.canvas.app.run()


if __name__ == "__main__":
    app = Application()

    try:
        app.run()
    except KeyboardInterrupt:
        pass

    app.shutdown()

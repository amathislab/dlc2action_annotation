#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in https://github.com/AlexEMG/DLC2action/LICENSE.AGPL.
#
import vispy
from vispy.scene import SceneCanvas
from ttf_opensans import opensans
from matplotlib.pyplot import cm
import numpy as np


class BoxVisual:
    def __init__(self, font_size, name, color, seq, cur, scene, display_name):
        self.color = color
        self.seq = seq
        self.display_name = display_name
        my_font = opensans()
        font = my_font.imagefont(size=font_size)
        self.font_x, self.font_y = font.getsize(name)
        self.font_x = self.font_x * 2
        self.font_y = self.font_y * 1.5
        self.seq[:, -2] += self.font_x / 2
        self.seq[:, -1] += self.font_y / 2
        c1, c2, w, h, rect_x, rect_y = self.seq[cur]
        self.box = vispy.scene.visuals.Rectangle(
            color=(0, 0, 0, 0),
            border_width=3,
            center=(c1, c2),
            width=w,
            height=h,
            border_color=color,
            parent=scene,
        )
        self.box.order = -1
        if self.display_name:
            self.rect = vispy.scene.visuals.Rectangle(
                color=color,
                center=(rect_x, rect_y),
                width=self.font_x,
                height=self.font_y,
                parent=scene,
            )
            self.text = vispy.scene.visuals.Text(
                name,
                pos=(rect_x, rect_y),
                color="white",
                font_size=font_size,
                parent=scene,
            )
        else:
            self.rect = vispy.scene.visuals.Rectangle(
                color=color,
                center=(-100, -100),
                width=self.font_x,
                height=self.font_y,
                parent=scene,
            )
            self.text = vispy.scene.visuals.Text(
                name, pos=(-100, -100), color="white", font_size=font_size, parent=scene
            )
        self.rect.order = -1
        self.text.order = -1

    def set_attrs(self, cur, hide=False):
        if hide:
            c1, c2, w, h, rect_x, rect_y = [-100, -100, 10, 10, -100, -100]
        else:
            c1, c2, w, h, rect_x, rect_y = self.seq[cur]
        self.box.center = (c1, c2)
        self.box.width = w
        self.box.height = h
        if self.display_name:
            self.rect.center = (rect_x, rect_y)
            self.text.pos = (rect_x, rect_y)

    def set_display_names(self, display_name):
        if display_name > 0:
            self.display_name = True
        else:
            self.display_name = False


class AnimalMarkers(vispy.scene.visuals.Markers):
    def __init__(
        self, vb, animal, edge_color, pos, color_order=None, rainbow=False, **kwargs
    ):
        super(AnimalMarkers, self).__init__(edge_color=edge_color, pos=pos, **kwargs)
        self.unfreeze()
        self.color = edge_color
        self.vb = vb
        self.animal = animal
        self.pos = pos
        self.rainbow = rainbow
        self.color_order = color_order

    def set_color(self, color):
        self.color = color

    def update_data(self, pos, al_mode, al_animal):
        size = self.vb.skeleton_size
        self.pos = pos
        if self.animal == self.vb.current_animal:
            face_color = self.color
            edge_color = self.color
        else:
            if al_mode:
                face_color = "grey"
                edge_color = "white"
                if self.animal == al_animal:
                    face_color = self.color
                    size = self.vb.skeleton_size + 1
            else:
                face_color = "white"
                edge_color = self.color
        if self.rainbow:
            face_color = cm.rainbow(np.linspace(0, 1, len(pos)))
            if self.color_order is not None and len(self.color_order) == len(pos):
                pos = pos[self.color_order]
                edge_color = None

        super(AnimalMarkers, self).set_data(
            pos=pos, edge_color=edge_color, face_color=face_color, size=size
        )


class Markers3d(vispy.scene.visuals.Markers):
    def __init__(self, vb, pos, rainbow=False, color_len=None, *args, **kwargs):
        super(Markers3d, self).__init__(pos=pos, spherical=True, *args, **kwargs)
        self.unfreeze()
        self.color = "white"
        self.vb = vb
        self.pos = pos
        self.set_gl_state("translucent", blend=True, depth_test=True)
        self.rainbow = rainbow
        self.color_len = color_len

    def set_color(self, color):
        self.color = color

    def update_data(self, pos):
        size = self.vb.skeleton_size
        if self.rainbow:
            if self.color_len is None:
                color_len = len(pos)
            else:
                color_len = self.color_len
            face_color = cm.rainbow(np.linspace(0, 1, color_len))[: len(pos)]
        else:
            face_color = self.color
        self.pos = pos
        super(Markers3d, self).set_data(pos=pos, face_color=face_color, size=size)

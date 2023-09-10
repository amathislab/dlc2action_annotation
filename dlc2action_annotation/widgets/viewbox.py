#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in https://github.com/AlexEMG/DLC2action/LICENSE.AGPL.
#
import pandas as pd
import vispy
from vispy.scene import SceneCanvas
import numpy as np
from collections import defaultdict
import warnings
from .visuals import BoxVisual, AnimalMarkers, Markers3d
from dlc2action_annotation.utils import SignalEmitter
from matplotlib.pyplot import cm
from copy import deepcopy


class VideoViewBox(vispy.scene.widgets.ViewBox):
    def __init__(
        self,
        n_ind,
        boxes,
        window,
        animals,
        points_df,
        segmentation,
        loaded,
        current,
        current_animal,
        video,
        node_transform,
        displayed_animals,
        skeleton_color,
        al_mode,
        al_animal,
        length,
        correct_mode,
        data_2d=None,
        skeleton=None,
        bodyparts_3d=None,
        *args,
        **kwargs
    ):
        super(VideoViewBox, self).__init__(*args, **kwargs)
        if skeleton is None:
            skeleton = []
        self.unfreeze()
        self.window = window
        self.node_transform_ = node_transform
        self.font_size = self.window.settings["font_size"]
        self.draw_boxes_bool = False
        self.selected_point = None
        self.skeleton_size = self.window.settings["skeleton_size"]
        self.backend = self.window.settings["backend"]
        self.box_update_freq = self.window.settings["detection_update_freq"]
        self.video = video
        self.animals = animals
        self.current_animal = current_animal
        self.points_df = points_df
        self.segmentation = segmentation
        self.loaded = loaded
        self.correct_mode = correct_mode
        self.emitter = SignalEmitter()
        self.current = current
        self.display_skeleton = len(skeleton) > 0
        self.display_repr = data_2d is not None
        self.length = length

        if self.video is not None:
            self.len = len(self.video)
        else:
            self.len = len(self.points_df)

        if points_df is not None:
            self.n_ind = len(self.animals)
            self.draw_points_bool = True
            self.bodyparts = points_df.names
        else:
            self.n_ind = n_ind
            self.draw_points_bool = False
            self.bodyparts = []

        self.color_order = None
        if bodyparts_3d is not None:
            if all([x in self.bodyparts for x in bodyparts_3d]):
                self.color_order = [self.bodyparts.index(x) for x in bodyparts_3d]
                for i, x in enumerate(self.bodyparts):
                    if x not in bodyparts_3d:
                        self.color_order.append(i)

        self.skeleton = []
        for a, b in skeleton:
            if a in self.bodyparts and b in self.bodyparts:
                self.skeleton.append([self.bodyparts.index(a), self.bodyparts.index(b)])
        self.skeleton = np.array(self.skeleton)

        self.data_2d = data_2d
        if self.points_df is not None:
            self.get_points(
                current, displayed_animals, skeleton_color, al_mode, al_animal
            )
        else:
            self.set_points()
        if boxes is not None:
            self.draw_boxes_bool = True
            self.boxes = boxes
            self.box_visuals = dict()
            for i in range(self.n_ind):
                seq = np.array([box[self.animals[i]] for box in self.boxes])
                self.box_visuals[i] = BoxVisual(
                    self.font_size,
                    self.animals[i],
                    skeleton_color(i),
                    seq,
                    current,
                    self.scene,
                    display_name=False,
                )
        else:
            self.box_visuals = {}

        # line_vis = vispy.scene.visuals.Line(pos=np.array([[0, 0], [200, 200], [100, 300]]), color='red', parent=self.scene)

    def initialize(self, current, mask_opacity):
        # noinspection PyTypeChecker
        self.camera = vispy.scene.cameras.PanZoomCamera(flip=(False, True))
        self.camera.aspect = 1
        if self.video is None:
            canvas_shape = self.window.settings["canvas_size"]
        else:
            canvas_shape = self.video.shape[1:-1]
        canvas_size = np.min(canvas_shape)
        scale = np.min([self.camera.rect.width, self.camera.rect.height])
        zoom = canvas_size / scale
        self.camera.zoom(zoom, center=((1 - canvas_shape[-1] / (zoom)) / 2, 0))

        if self.video is not None:
            self.image = vispy.scene.visuals.Image(
                data=self.video[current - self.loaded[0]], parent=self.scene, order=0
            )
            if self.segmentation is not None:
                self.mask = vispy.scene.visuals.Image(
                    data=self.segmentation.get_mask(current - self.loaded[0]),
                    parent=self.scene,
                    order=1,
                )
                self.mask.opacity = mask_opacity
                self.mask.set_gl_state("translucent", depth_test=False, blend=True)

    def set_image(self, current, displayed_animals, skeleton_color, al_mode, al_animal):
        index = current - self.loaded[0]
        if self.video is not None and not (index < 0 or index >= len(self.video)):
            self.image.set_data(self.video[index])
            if self.segmentation is not None:
                self.mask.set_data(self.segmentation.get_mask(index))
        if (
            self.draw_points_bool or self.display_repr
        ) and current % self.box_update_freq == 0:
            self.draw_points(
                current,
                displayed_animals,
                skeleton_color,
                al_mode,
                al_animal,
                hide=False,
            )

    def get_image(self, current):
        index = current - self.loaded[0]
        if self.video is not None and not (index < 0 or index >= len(self.video)):
            return self.video[index]
        else:
            return None

    def get_points(
        self, current, displayed_animals, skeleton_color, al_mode, al_animal
    ):
        self.points = defaultdict(
            lambda: AnimalMarkers(
                vb=self,
                animal=self.animals[0],
                pos=np.array([[0, 0]]),
                parent=self.scene,
                size=self.skeleton_size,
                edge_color="white",
                face_color="white",
                color_order=self.color_order,
            )
        )
        self.skeletons = defaultdict(
            lambda: vispy.scene.visuals.Line(
                pos=np.array([[0, 0]]),
                parent=self.scene,
            )
        )
        for animal in displayed_animals:
            i = self.animals.index(animal)
            color = skeleton_color(i)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="indexing")
                vis = AnimalMarkers(
                    vb=self,
                    animal=animal,
                    pos=self.points_df.get_coord(current, animal),
                    parent=self.scene,
                    size=self.skeleton_size,
                    edge_color=color,
                    face_color="white",
                    color_order=self.color_order,
                )
                vis.update_data(
                    self.points_df.get_coord(current, animal), al_mode, al_animal
                )

            vis.order = -1
            self.points[animal] = vis
            self.current = current
            if len(self.skeleton) > 0:
                pos = self.points_df.get_coord(current, animal)
                skeleton = np.array(
                    [
                        x
                        for x in self.skeleton
                        if np.sum(pos[x[0]]) > 0 and np.sum(pos[x[1]]) > 0
                    ]
                )
                if len(skeleton) == 0:
                    skeleton = np.array([[0, 0]])
                skeleton_vis = vispy.scene.visuals.Line(
                    pos=pos, color="cornflowerblue", parent=self.scene, connect=skeleton
                )
                self.skeletons[animal] = skeleton_vis
                skeleton_vis.order = -1
        self.set_repr()

    def set_repr(self):
        if self.data_2d is not None:
            self.points_2d = vispy.scene.visuals.Markers(
                pos=self.data_2d[0],
                parent=self.scene,
                face_color="white",
                edge_color="black",
                symbol="cross",
                size=self.skeleton_size + 2,
            )
        else:
            self.points_2d = vispy.scene.visuals.Markers(
                pos=np.array([[0, 0]]),
                parent=self.scene,
                face_color="white",
                edge_color="black",
                size=self.skeleton_size + 2,
            )

    def set_points(self):
        self.points = {
            ind: AnimalMarkers(
                vb=self,
                animal=self.animals[0],
                pos=np.array([[0, 0]]),
                parent=self.scene,
                edge_color="white",
                size=0,
            )
            for i, ind in enumerate(self.animals)
        }
        self.skeletons = {
            ind: vispy.scene.visuals.Line(pos=np.array([[0, 0]]), parent=self.scene)
            for ind in self.animals
        }
        self.set_repr()

    def set_current_animal(self, animal):
        if animal in self.animals:
            if animal != self.current_animal:
                self.emitter.animal_changed.emit()
            self.current_animal = animal

    def draw_points(
        self, current, displayed_animals, skeleton_color, al_mode, al_animal, hide=False
    ):
        to_pop = []
        for ind in self.points.keys():
            if ind not in displayed_animals:
                to_pop.append(ind)
        for ind in to_pop:
            self.points[ind].parent = None
            self.points.pop(ind)
        for ind in displayed_animals:
            skeleton_vis = self.skeletons[ind]
            if ind not in self.points.keys():
                vis = self.points[ind]
                vis.animal = ind
                vis.set_color(skeleton_color(self.animals.index(ind)))
            else:
                vis = self.points[ind]
            skeleton_vis.order = -1
            vis.order = -1
            if hide:
                vis.update_data(np.array([[0, 0]]), al_mode, al_animal)
                skeleton_vis.set_data(np.array([[0, 0]]))
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="indexing")
                        pos = self.points_df.get_coord(current, vis.animal)
                        vis.update_data(
                            pos,
                            al_mode,
                            al_animal,
                        )
                        if self.display_skeleton:
                            skeleton = np.array(
                                [
                                    x
                                    for x in self.skeleton
                                    if np.sum(pos[x[0]]) > 0 and np.sum(pos[x[1]]) > 0
                                ]
                            )
                            if len(skeleton) == 0:
                                skeleton = np.array([[0, 0]])
                            skeleton_vis.set_data(pos=pos, connect=skeleton)
                        else:
                            skeleton_vis.set_data(
                                pos=np.array([[0, 0]]), connect="segments"
                            )
                except:
                    vis.update_data(np.array([[0, 0]]), al_mode, al_animal)
                    skeleton_vis.set_data(pos=np.array([[0, 0]]), connect="segments")
        if self.skeleton_size == 0:
            size = 0
        else:
            size = self.skeleton_size + 3
        if self.data_2d is not None:
            if self.display_repr:
                self.points_2d.set_data(pos=self.data_2d[current], size=size)
            else:
                self.points_2d.set_data(np.array([[0, 0]]))
        self.current = current

    def draw_boxes(self, current, hide=False):
        for i in range(self.n_ind):
            self.box_visuals[i].set_attrs(current, hide)

    def set_display_names(self, state):
        for i in range(self.n_ind):
            self.box_visuals[i].set_display_names(state)

    def select(self, event):
        tr = self.node_transform_(self.points[0])
        pos = tr.map(event.pos)[:2]
        for animal in self.points:
            norm = np.linalg.norm(self.points[animal].pos - pos, axis=1)
            if np.sum(norm < 5) > 0:
                # if self.correct_mode:
                p = np.where(norm < 5)[0][0]
                self.emitter.hovered.emit(self.bodyparts[int(p)])
                if self.correct_mode:
                    self.selected_point = (animal, p)
                    self.camera.interactive = False
                return animal
                # else:
                #     return animal
        return None

    def on_mouse_press(self, event):
        animal = self.select(event)
        if animal is not None:
            self.emitter.animal_clicked.emit(animal)

    def on_mouse_move(self, event):
        if self.correct_mode:
            tr = self.node_transform_(self.image)
            pos = tr.map(event.pos)[:2]
            if self.selected_point is not None:
                animal, point = self.selected_point
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="indexing")
                    self.points_df.set_coord(self.current, animal, point, pos[:2])
                    self.emitter.points_changed.emit()

    def on_mouse_release(self, event):
        self.emitter.point_released.emit()
        if self.selected_point is not None:
            animal, _ = self.selected_point
            self.camera.interactive = True
            self.selected_point = None

    def set_size(self, value):
        self.skeleton_size = value
        for k in self.points:
            self.points[k].set_data(size=value)
        self.points_2d.set_data(size=value)

    def get_keypoints(self, current):
        corrections = {current: {}}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="indexing")
            for animal in self.animals:
                corrections[current][animal] = self.points_df.get_coord(current, animal)
        return corrections

    def switch_skeleton(self):
        if len(self.skeleton) > 0:
            self.display_skeleton = not self.display_skeleton
        if not self.display_skeleton:
            for ind in self.skeletons:
                self.skeletons[ind].set_data(pos=np.array([[0, 0]]))

    def switch_repr(self):
        if self.data_2d is not None:
            self.display_repr = not self.display_repr
        if not self.display_repr:
            self.points_2d.set_data(np.array([[0, 0]]))

    def switch_rainbow(self):
        for key, vis in self.points.items():
            self.points[key].rainbow = not vis.rainbow

    def get_ind_start_end(self, animal):
        if self.points_df is None:
            return 0, self.length
        return self.points_df.get_start_end(animal)


class VideoViewBox3D(vispy.scene.widgets.ViewBox):
    def __init__(
        self, data_3d, parent, skeleton_size, skeleton, bodyparts, length, color_len=None
    ):
        super(VideoViewBox3D, self).__init__(parent=parent)
        self.unfreeze()
        self.data = data_3d
        self.skeleton_size = skeleton_size
        self.color_len = color_len
        self.skeleton = []
        self.length = length
        if bodyparts is not None:
            for a, b in skeleton:
                if a in bodyparts and b in bodyparts:
                    self.skeleton.append([bodyparts.index(a), bodyparts.index(b)])
            self.skeleton = np.array(self.skeleton)
        self.display_skeleton = len(self.skeleton) > 0

    def initialize(self, current):
        # noinspection PyTypeChecker
        self.visual = Markers3d(
            self, self.data[current], parent=self.scene, color_len=self.color_len
        )
        self.skeleton_visual = vispy.scene.visuals.Line(
            pos=self.data[current],
            color="cornflowerblue",
            parent=self.scene,
            connect=self.skeleton,
        )
        self.skeleton_visual.order = -1
        self.camera = vispy.scene.cameras.PanZoomCamera(flip=(False, True))
        self.camera.aspect = 1

    def set_image(self, current, *args, **kwargs):
        self.visual.update_data(self.data[current])
        self.visual.order = -1
        if self.display_skeleton:
            self.skeleton_visual.set_data(pos=self.data[current], connect=self.skeleton)
        else:
            self.skeleton_visual.set_data(pos=np.array([[0, 0]]), connect="segments")

    def set_size(self, value):
        self.visual.skeleton_size = value
        self.visual.set_data(size=value)

    def set_display_names(self, state):
        pass

    def set_current_animal(self, animal):
        pass

    def switch_rainbow(self):
        self.visual.rainbow = not self.visual.rainbow

    def switch_skeleton(self):
        if len(self.skeleton) > 0:
            self.display_skeleton = not self.display_skeleton

    def get_ind_start_end(self, animal):
        return 0, self.length



from PyQt5.QtCore import QTimer
from vispy.scene import SceneCanvas
from utils import WorkerThread, read_stack, SignalEmitter, get_2d_files
from .viewbox import VideoViewBox, VideoViewBox3D


class VideoCanvas(SceneCanvas):
    def __init__(
        self,
        window,
        stacks,
        shapes,
        lengths,
        n_ind,
        animals,
        boxes,
        points_df_list,
        segmentation_list,
        index_dict,
        current,
        current_animal,
        correct_mode,
        al_points,
        mask_opacity,
        data_3d,
        data_2d,
        skeleton,
        bodyparts_3d,
    ):
        super(VideoCanvas, self).__init__()
        if skeleton is None:
            skeleton = []
        self.unfreeze()
        self.window = window
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_videos)
        self.emitter = SignalEmitter()
        self.al_points = al_points
        self.data_3d = data_3d
        self.data_2d = data_2d
        self.skeleton = skeleton
        self.bodyparts_3d = bodyparts_3d
        self.start(
            stacks,
            shapes,
            lengths,
            n_ind,
            animals,
            boxes,
            points_df_list,
            segmentation_list,
            index_dict,
            current,
            current_animal,
            correct_mode,
            mask_opacity,
        )

    def change_al_mode(self, value):
        self.worker.stop()
        if value:
            self.loading_al_ind = self.window.cur_al_point
            self.max_len = self.window.settings["max_loaded_frames_al"]
            self.load_chunk = self.window.settings["load_chunk_al"]
            self.buffer = self.window.settings["load_buffer_al"]
            self.loaded = [
                [
                    max(start - self.window.al_buffer, 0),
                    max(start - self.window.al_buffer, 0) + 1,
                ]
                for start, end, ind in self.al_points
            ]
            self.videos = [
                [
                    read_stack(stack, start, end, shape, self.backend)
                    for stack, shape in zip(self.stacks, self.shapes)
                ]
                for start, end in self.loaded
            ]
        else:
            self.loading_al_ind = None
            self.max_len = self.window.settings["max_loaded_frames"]
            self.load_chunk = self.window.settings["load_chunk"]
            self.buffer = self.window.settings["load_buffer"]
            self.loaded = [self.current, self.current + 1]
            self.videos = [
                read_stack(stack, self.loaded[0], self.loaded[1], shape, self.backend)
                for stack, shape in zip(self.stacks, self.shapes)
            ]
        self.loading_status = False
        self.emitter.mode_changed.emit(value)

    def start(
        self,
        stacks,
        shapes,
        lengths,
        n_ind,
        animals,
        boxes,
        points_df_list,
        segmentation_list,
        index_dict,
        current,
        current_animal,
        correct_mode,
        mask_opacity,
    ):
        self.start = 0
        self.end = 200
        self.loading_status = False
        self.n_ind = n_ind
        self.al_mode_changed = False
        self.stacks, self.shapes, self.lens = stacks, shapes, lengths
        self.speed = self.window.settings["default_frequency"]
        self.backend = self.window.settings["backend"]
        self.al_window_num = self.window.settings["al_window_num"]
        if self.window.al_mode:
            self.loading_al_ind = self.window.cur_al_point
            self.loaded = [
                [
                    max(cur - self.window.al_buffer, 0),
                    max(cur - self.window.al_buffer, 0) + 1,
                ]
                for cur, end, ind in self.window.al_points
            ]
            self.max_len = self.window.settings["max_loaded_frames_al"]
            self.load_chunk = self.window.settings["load_chunk_al"]
            self.buffer = self.window.settings["load_buffer_al"]
            self.videos = [
                [
                    read_stack(stack, start, end, shape, self.backend)
                    for stack, shape in zip(self.stacks, self.shapes)
                ]
                for start, end in self.loaded
            ]
        else:
            self.loaded = [current, current + 1]
            self.max_len = self.window.settings["max_loaded_frames"]
            self.load_chunk = self.window.settings["load_chunk"]
            self.buffer = self.window.settings["load_buffer"]
            self.videos = [
                read_stack(stack, self.loaded[0], self.loaded[1], shape, self.backend)
                for stack, shape in zip(self.stacks, self.shapes)
            ]
        if self.videos[0] is not None:
            self.len_global = lengths[0]
        else:
            self.len_global = len(points_df_list[0])
            self.loaded = [0, self.len_global]

        self.current = current

        self.play = False
        self.next_bunch = None
        self.handled = False
        self.animals = animals
        self.index_dict = index_dict

        if current_animal is not None:
            self.current_animal = current_animal
        else:
            self.current_animal = animals[0]

        self.set_grid(
            boxes, points_df_list, segmentation_list, correct_mode, mask_opacity
        )
        if self.videos[0] is not None:
            self.load_data()

    def draw_points(self):
        for vb in self.viewboxes:
            if isinstance(vb, VideoViewBox):
                vb.draw_points(
                    self.current,
                    self.window.displayed_animals,
                    self.window.skeleton_color,
                    self.window.al_mode,
                    self.window.al_animal,
                )

    def hide_boxes(self):
        for vb in self.viewboxes:
            if isinstance(vb, VideoViewBox):
                vb.draw_points(
                    self.current,
                    self.window.displayed_animals,
                    self.window.skeleton_color,
                    self.window.al_mode,
                    self.window.al_animal,
                    hide=True,
                )
                # vb.draw_boxes(hide=True)
                vb.draw_points_bool = False
                # vb.draw_boxes_bool = False

    def set_box_update(self, value):
        for vb in self.viewboxes:
            if isinstance(vb, VideoViewBox):
                vb.box_update_freq = value
                vb.draw_points_bool = True
                # vb.draw_boxes_bool = True

    def set_current_frame(self, value, center=False):
        if self.window.al_mode:
            loaded = self.loaded[self.window.cur_al_point]
        else:
            loaded = self.loaded
        next = value % self.len_global
        action_needed = not self.handled or next - self.current not in [0, 1]
        if action_needed:
            for vb in self.viewboxes:
                vb.current = next
                self.current = next
            self.window.set_edges(center)
        if next < loaded[0]:
            if action_needed:
                if loaded[0] - self.load_chunk < next:
                    self.next_bunch = {
                        "start": next,
                        "end": loaded[0],
                        "al_ind": self.window.cur_al_point,
                    }
                else:
                    self.next_bunch = {
                        "start": next,
                        "al_ind": self.window.cur_al_point,
                    }
            if not self.loading_status:
                self.start_loading()
            self.handled = True
        elif next >= loaded[1]:
            if action_needed:
                if loaded[1] + self.load_chunk <= next:
                    self.next_bunch = {
                        "start": next,
                        "al_ind": self.window.cur_al_point,
                    }
                else:
                    self.next_bunch = {
                        "start": loaded[1],
                        "al_ind": self.window.cur_al_point,
                    }
            if not self.loading_status:
                self.start_loading()
            self.handled = True
        else:
            self.handled = False
            for vb in self.viewboxes:
                vb.set_image(
                    self.current,
                    self.window.displayed_animals,
                    self.window.skeleton_color,
                    self.window.al_mode,
                    self.window.al_animal,
                )
            if not self.loading_status and self.current > loaded[0] + self.buffer:
                self.load_data()
        if self.index_dict[self.current] is not None:
            self.window.change_displayed_animals(self.index_dict[self.current])

    def start_loading(self):
        if self.next_bunch is not None:
            self.load_data(**self.next_bunch)
            self.next_bunch = None
        else:
            self.load_data()

    def set_al_point(self, al_point):
        long = [i for i in range(len(self.videos)) if self.videos[i][0].shape[0] > 1]
        if (
            len(long) > 0
            and al_point - long[0] > 1
            and len(long) >= self.al_window_num
            and long[-1] + 1 < len(self.videos)
        ):
            for i in range(len(self.videos[long[0]])):
                self.videos[long[0]][i] = self.videos[long[0]][i][:1]
            self.loaded[long[0]] = [
                self.loaded[long[0]][0],
                self.loaded[long[0]][0] + 1,
            ]
            long.pop(0)
            self.next_bunch = {
                "start": self.window.al_points[al_point][0],
                "al_ind": long[-1] + 1,
            }
            self.start_loading()
        for vb, video in zip(self.viewboxes, self.videos[al_point]):
            vb.set_image(
                self.current,
                self.window.displayed_animals,
                self.window.skeleton_color,
                self.window.al_mode,
                self.window.al_animal,
            )
            if isinstance(vb, VideoViewBox):
                vb.video = video
                vb.loaded = self.loaded[al_point]

    def set_display_names(self, state):
        for vb in self.viewboxes:
            vb.set_display_names(state)

    def set_play(self, value=None):
        if value in [True, False]:
            self.play = value
        else:
            self.play = not self.play
        if self.play:
            self.timer.start(self.speed)
        else:
            self.timer.stop()

    def play_videos(self):
        self.set_current_frame(self.current + 1)

    def change_speed(self, value):
        self.speed = value
        self.timer.stop()
        self.set_play(self.play)

    def load_data(self, start=None, end=None, al_ind=None):
        if self.videos[0] is None:
            return
        if self.window.al_mode and al_ind is None:
            al_ind = self.window.cur_al_point
        if self.window.al_mode:
            self.loading_al_ind = al_ind
            loaded = self.loaded[al_ind]
            videos = self.videos[al_ind]
        else:
            loaded = self.loaded
            videos = self.videos
        self.loading_status = True
        if start is None or (start < loaded[1] and start > loaded[0]):
            start = loaded[1]
        if end is None or (end > loaded[0] and end < loaded[1]):
            if len(videos[0]) + self.load_chunk >= self.max_len:
                extra = max(self.current - loaded[0] - self.buffer, 0)
                end = min(loaded[0] + self.max_len + extra, start + self.load_chunk)
            else:
                end = start + self.load_chunk
        if end > self.len_global:
            end = self.len_global
        if start < 0:
            start = 0
        if start >= end:
            self.loading_status = False
            return
        self.loading = (start, end)
        try:
            self.worker.stop()
        except:
            pass
        self.worker = WorkerThread(
            self.stacks,
            self.shapes,
            videos,
            self.loading,
            loaded,
            self.max_len,
            self.current,
            self.buffer,
            self.backend,
        )
        self.worker.job_done.connect(self.on_data_received)
        self.worker.start()

    def receive_data_normal(self, data):
        self.videos, self.loaded = data
        for vb, video in zip(self.viewboxes, self.videos):
            vb.set_image(
                self.current,
                self.window.displayed_animals,
                self.window.skeleton_color,
                self.window.al_mode,
                self.window.al_animal,
            )
            if isinstance(vb, VideoViewBox):
                vb.video = video
                vb.loaded = self.loaded
        if self.next_bunch is not None:
            self.load_data(**self.next_bunch)
            self.next_bunch = None
        elif (
            self.current >= self.loaded[0]
            and self.current <= self.loaded[1]
            and self.loaded[1] < self.len_global
        ):
            if (
                self.videos[0].shape[0] >= self.max_len
                and self.current < self.loaded[0] + self.buffer
            ):
                self.loading_status = False
            else:
                self.load_data()
        else:
            self.loading_status = False

    def receive_data_al(self, data):
        cur_ind = self.window.cur_al_point
        loading_ind = self.loading_al_ind
        videos, loaded = data
        self.videos[loading_ind] = videos
        self.loaded[loading_ind] = loaded
        long = [i for i in range(len(self.videos)) if self.videos[i][0].shape[0] > 1]
        # print(f'{long=}, {cur_ind=}')
        if cur_ind - long[0] > 1 and len(long) > self.al_window_num:
            # print('case 1')
            for i in range(len(self.videos[long[0]])):
                self.videos[long[0]][i] = self.videos[long[0]][i][:1]
            self.loaded[long[0]] = [
                self.loaded[long[0]][0],
                self.loaded[long[0]][0] + 1,
            ]
            long.pop(0)
        long_end = long[0] + self.al_window_num
        for vb, video in zip(self.viewboxes, self.videos[self.window.cur_al_point]):
            vb.set_image(
                self.current,
                self.window.displayed_animals,
                self.window.skeleton_color,
                self.window.al_mode,
                self.window.al_animal,
            )
            if isinstance(vb, VideoViewBox):
                vb.video = video
                vb.loaded = self.loaded[self.window.cur_al_point]
        if self.next_bunch is not None:
            # print(f'{self.next_bunch=}')
            self.load_data(**self.next_bunch)
            self.next_bunch = None
        elif (
            self.current >= self.loaded[cur_ind][0]
            and self.current <= self.loaded[cur_ind][1]
            and self.loaded[cur_ind][1] < self.len_global
        ):
            if (
                self.videos[cur_ind][0].shape[0] < self.max_len
                or self.current > self.loaded[cur_ind][0] + self.buffer
            ):
                # print('case 2')
                self.load_data(al_ind=cur_ind)
            else:
                # print('case 3')
                self.loading_status = False
                end = min(long_end, len(self.videos))
                for loading_ind in range(loading_ind, end):
                    if self.videos[loading_ind][0].shape[0] < self.max_len:
                        self.load_data(al_ind=loading_ind)
                        break
        else:
            self.loading_status = False

    def on_data_received(self, data):
        if self.window.al_mode:
            self.receive_data_al(data)
        else:
            self.receive_data_normal(data)

    def set_grid(
        self, box_list, points_df_list, segmentation_list, correct_mode, mask_opacity
    ):
        self.grid = self.central_widget.add_grid(spacing=10)
        l = len(self.stacks)
        if self.data_3d is not None:
            l += 1
        if l == 1:
            self.grid_y = 1
            self.grid_x = 1
        elif l == 2:
            self.grid_y = 1
            self.grid_x = 2
        elif l == 3 or l == 4:
            self.grid_y = 2
            self.grid_x = 2
        elif l == 5 or l == 6:
            self.grid_y = 2
            self.grid_x = 3
        elif l in [7, 8, 9]:
            self.grid_y = 3
            self.grid_x = 3
        else:
            raise ("Too many videos!")

        if self.window.al_mode:
            loaded = self.loaded[self.window.cur_al_point]
            videos = self.videos[self.window.cur_al_point]
        else:
            loaded = self.loaded
            videos = self.videos

        for i in range(self.grid_y):
            for j in range(self.grid_x):
                n = i * self.grid_x + j
                for x in [points_df_list, box_list]:
                    while len(x) < len(videos):
                        x.append(None)
                if n < len(videos):
                    vb = VideoViewBox(
                        self.n_ind,
                        box_list[n],
                        self.window,
                        self.animals,
                        points_df_list[n],
                        segmentation_list[n],
                        loaded,
                        self.current,
                        self.current_animal,
                        videos[n],
                        displayed_animals=self.window.displayed_animals,
                        skeleton_color=self.window.skeleton_color,
                        al_mode=self.window.al_mode,
                        parent=self.scene,
                        node_transform=self.scene.node_transform,
                        al_animal=self.window.al_animal,
                        correct_mode=correct_mode,
                        data_2d=self.data_2d[n],
                        skeleton=self.skeleton,
                        bodyparts_3d=self.bodyparts_3d,
                        length=self.len_global,
                    )
                    self.grid.add_widget(vb, i, j)
                    vb.initialize(self.current, mask_opacity)
                    vb.emitter.animal_clicked.connect(self.window.set_animal)
                    vb.emitter.animal_changed.connect(self.emitter.animal_changed.emit)
                    vb.emitter.points_changed.connect(self.redraw)
                    vb.emitter.hovered.connect(self.emitter.hovered.emit)
                    vb.emitter.point_released.connect(self.emitter.point_released.emit)

                    # if n > 0:
                    #     self.grid.children[0].camera.link(vb.camera)
                elif n == len(videos) and self.data_3d is not None:
                    color_len = None
                    for x in points_df_list:
                        if x is not None:
                            color_len = len(x.names)
                            break
                    vb = VideoViewBox3D(
                        self.data_3d,
                        parent=self.scene,
                        skeleton_size=self.window.settings["skeleton_size"],
                        color_len=color_len,
                        bodyparts=self.window.settings["3d_bodyparts"],
                        skeleton=self.skeleton,
                        length=self.len_global,
                    )
                    vb.initialize(self.current)
                    vb.camera = "turntable"
                    vb.camera.fov = 45
                    vb.camera.distance = 75
                    self.grid.add_widget(vb, i, j)

        self.viewboxes = self.grid.children

    def redraw(self):
        self.draw_points()
        self.update()

    def set_current_animal(self, animal):
        self.current_animal = animal
        for vb in self.viewboxes:
            vb.set_current_animal(animal)

    def on_times_change(self, start=None, end=None):
        if start:
            self.start = start
        if end:
            self.end = end
        self.window.bar.frames = self.end - self.start

    def set_cat_id(self, cat_id):
        for vb in self.viewboxes:
            if isinstance(vb, VideoViewBox) and vb.segmentation is not None:
                vb.segmentation.cat_id = cat_id

    def get_segmentation_cats(self):
        cats = set()
        for vb in self.viewboxes:
            if isinstance(vb, VideoViewBox) and vb.segmentation is not None:
                cats.update(vb.segmentation.cats)
        return sorted(list(cats))

    def set_mask_opacity(self, value):
        for vb in self.viewboxes:
            if isinstance(vb, VideoViewBox) and vb.segmentation is not None:
                vb.mask.opacity = value

    def switch_rainbow(self):
        for vb in self.viewboxes:
            vb.switch_rainbow()

    def switch_skeleton(self):
        for vb in self.viewboxes:
            vb.switch_skeleton()

    def switch_repr(self):
        for vb in self.viewboxes:
            if isinstance(vb, VideoViewBox):
                vb.switch_repr()

    def on_key_press(self, event):
        if event.key.name == "Space":
            self.window.on_play()
        elif event.key.name == "Right":
            self.window.next()
        elif event.key.name == "Left":
            self.window.prev()
        elif event.key.name == "Escape":
            self.window.on_escape()
        elif event.key.name == "Enter":
            self.window.on_enter()
        elif event.key.name == "-":
            self.window.on_minus()
        elif event.key.name == "=":
            self.window.on_plus()
        elif event.key.name == "Z":
            if "Control" in [key.name for key in event.modifiers]:
                self.window.on_z()
        elif event.key.name in self.window.active_shortcuts():
            self.window.on_shortcut(event.key.name)
        elif event.key.name in list(map(str, range(min(self.window.n_animals(), 10)))):
            self.window.set_animal(int(event.key.name))
        else:
            print(f"canvas didn't recognise key {event.key.name}")

    def get_ind_start_end(self, animal):
        starts = []
        ends = []
        for vb in self.viewboxes:
            start, end = vb.get_ind_start_end(animal)
            if start is not None:
                starts.append(start)
            if end is not None:
                ends.append(end)
        if len(starts) == 0:
            start = None
        else:
            start = min(starts)
        if len(ends) == 0:
            end = None
        else:
            end = max(ends)
        return start, end

    # def on_mouse_press(self, event):
    #     tr = self.scene.node_transform(self.line)
    #     pos = tr.map(event.pos)

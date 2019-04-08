import numpy as np
import mujoco_py
from mujoco_py import MjSim
from mujoco_py.mjviewer import MjViewerBasic


def add_selection_logger(viewer: MjViewerBasic, sim: MjSim, callback=None):
    """
    Adds a click handler that prints information about the body clicked with the middle mouse button.
    Make sure to call env.render() so that a viewer exists before calling this function.
    :param viewer: the MuJoCo viewer in use.
    :param sim: the MuJoCo simulation object.
    :param callback: optional callback to be called when the user clicks.
    """

    import glfw

    def mouse_callback(window, button, act, mods):
        viewer._mouse_button_callback(window, button, act, mods)
        middle_pressed = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        viewer.pert.active = 0
        if middle_pressed:
            w, h = glfw.get_window_size(window)
            aspect_ratio = w / h
            x, y = viewer._last_mouse_x, viewer._last_mouse_y
            sel_point = np.zeros(3)
            res = mujoco_py.functions.mjv_select(sim.model, sim.data, viewer.vopt,
                                                 aspect_ratio, x/w, (h-y)/h, viewer.scn, sel_point)
            sel_body, sel_geom = '?', '?'
            if res != -1:
                sel_body = sim.model.body_id2name(sim.model.geom_bodyid[res])
                sel_geom = sim.model.geom_id2name(res)
            print(f'Selected {sel_body} ({sel_geom}) at {sel_point}')

            if callable(callback):
                callback(sel_point, x/w, (h-y)/h)

    def cursor_pos_callback(window, xpos, ypos):
        viewer._cursor_pos_callback(window, xpos, ypos)
        middle_pressed = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        if middle_pressed:
            w, h = glfw.get_window_size(window)
            aspect_ratio = w / h
            x, y = xpos, ypos
            sel_point = np.zeros(3)
            res = mujoco_py.functions.mjv_select(sim.model, sim.data, viewer.vopt,
                                                 aspect_ratio, x/w, (h-y)/h, viewer.scn, sel_point)
            sel_body, sel_geom = '?', '?'
            if res != -1:
                sel_body = sim.model.body_id2name(sim.model.geom_bodyid[res])
                sel_geom = sim.model.geom_id2name(res)
            print(f'Selected {sel_body} ({sel_geom}) at {sel_point}')

            if callable(callback):
                callback(sel_point, x/w, (h-y)/h)

    glfw.set_cursor_pos_callback(viewer.window, cursor_pos_callback)
    glfw.set_mouse_button_callback(viewer.window, mouse_callback)


def add_scroll_callback(viewer: MjViewerBasic, callback):

    assert callable(callback)
    import glfw

    def scroll_callback(window, x_offset, y_offset):
        mod_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or \
                    glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        if mod_shift:
            scroll_callback.pos -= 0.05 * y_offset
            callback(scroll_callback.pos)
        else:
            viewer._scroll_callback(window, x_offset, y_offset)

    scroll_callback.pos = 0.0
    glfw.set_scroll_callback(viewer.window, scroll_callback)

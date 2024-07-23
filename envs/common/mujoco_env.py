import os
import numpy as np
import mujoco
import mujoco_viewer

DEFAULT_SIZE = 500

class MujocoEnv():
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, sim_dt, control_dt):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            raise Exception("Provide full path to robot description package.")
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # set frame skip and sim dt
        self.frame_skip = (control_dt/sim_dt)
        self.model.opt.timestep = sim_dt

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        根据顺序。每行代码作用为：
        1. 设置摄像机跟踪的物体 ID。这里设置为 1，通常意味着跟踪模型中的第二个物体（因为索引从 0 开始）。
        2. 设置摄像机距离。self.model.stat.extent 可能是模型的整体尺寸，这里将距离设置为模型尺寸的 1.5 倍。
        3. 设置摄像机的观察点。这里调整了 Z 轴（高度）和 X 轴的位置。
        4。设置摄像机的仰角，-20 度表示略微向下看。
        5。设置几何体组的可见性。这里可能是确保某个特定组的几何体是可见的。
        6。设置每一帧都渲染，这可能会影响性能但提高了视觉更新的频率。
        """
        # self.viewer.cam.trackbodyid = 1
        # self.viewer.cam.distance = self.model.stat.extent * 1.5
        # self.viewer.cam.lookat[2] = 1.5
        # self.viewer.cam.lookat[0] = 2.0
        # self.viewer.cam.elevation = -20
        # self.viewer.vopt.geomgroup[0] = 1
        # self.viewer._render_every_frame = True
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.azimuth = 180
        self.viewer.cam.distance = self.model.stat.extent * 2.0
        self.viewer.cam.lookat[2] = 0.6
        self.viewer.cam.lookat[0] = 0.1
        self.viewer.cam.elevation = -30
        self.viewer.vopt.geomgroup[0] = 1
        self.viewer._render_every_frame = True


    def viewer_is_paused(self):
        return self.viewer._paused

    # -----------------------------

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer_setup()
        self.viewer.render()

    def uploadGPU(self, hfieldid=None, meshid=None, texid=None):
        # hfield
        if hfieldid is not None:
            mujoco.mjr_uploadHField(self.model, self.viewer.ctx, hfieldid)
        # mesh
        if meshid is not None:
            mujoco.mjr_uploadMesh(self.model, self.viewer.ctx, meshid)
        # texture
        if texid is not None:
            mujoco.mjr_uploadTexture(self.model, self.viewer.ctx, texid)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

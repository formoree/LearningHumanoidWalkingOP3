import os
import numpy as np
import transforms3d as tf3
import collections

from tasks import standing_task
from envs.common import mujoco_env
from envs.common import robot_interface
from envs.op3 import robot_stand

from .gen_xml_stand import builder


class Op3StandEnv(mujoco_env.MujocoEnv):
    def __init__(self):
        sim_dt = 0.0025
        control_dt = 0.025
        frame_skip = (control_dt/sim_dt)
        # 大道至简，尽量不对原model改动
        path_to_xml_out = '/tmp/mjcf-export/op3_stand/op3.xml'
        if not os.path.exists(path_to_xml_out):
            builder(path_to_xml_out)
        mujoco_env.MujocoEnv.__init__(self, path_to_xml_out, sim_dt, control_dt)
        """
        ['head_pan_act', 'head_tilt_act', 'l_ank_pitch_act', 'l_ank_roll_act', 'l_el_act', 'l_hip_pitch_act', 
        'l_hip_roll_act', 'l_hip_yaw_act', 'l_knee_act', 'l_sho_pitch_act', 'l_sho_roll_act', 'r_ank_pitch_act', 
        'r_ank_roll_act', 'r_el_act', 'r_hip_pitch_act', 'r_hip_roll_act', 'r_hip_yaw_act', 'r_knee_act', 
        'r_sho_pitch_act', 'r_sho_roll_act']
        """
        pdgains = np.zeros((20, 2))
        coeff = 1
        pdgains.T[0] = coeff * np.array([5, 5, 50, 50, 10, 50, 50, 50,
                                         50, 10, 10, 50, 50, 10, 50, 50, 50, 50, 10, 10, ])
        pdgains.T[1] = coeff * np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,])

        # list of desired actuators
        # 'r_hip_yaw', 'r_hip_roll', 'r_hip_pitch', 'r_knee', 'r_ank_pitch', 'r_ank_roll'
        # 'l_hip_yaw', 'l_hip_roll', 'l_hip_pitch', 'l_knee', 'l_ank_pitch', 'l_ank_roll'
        self.actuators = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                          11, 12, 13, 14, 15, 16, 17, 18, 19,]

        # set up interface
        self.interface = robot_interface.RobotInterface(self.model, self.data, 'r_ank_pitch_link', 'l_ank_pitch_link', 'body_link')
        # set up task
        self.task = standing_task.StandingTask(client=self.interface,
                                             dt=control_dt,
                                             neutral_foot_orient=np.array([1, 0, 0, 0]),
                                             root_body='body_link',
                                             lfoot_body='l_ank_pitch_link',
                                             rfoot_body='r_ank_pitch_link',
                                             head_body='body_link',
        )
        # set goal height
        self.task._goal_height_ref = 0.31
        self.task._total_duration = 1.1
        self.task._swing_duration = 0.75
        self.task._stance_duration = 0.35
        # call reset
        self.task.reset()

        self.robot = robot_stand.JVRC(pdgains.T, control_dt, self.actuators, self.interface)

        # define indices for action and obs mirror fns
        base_mir_obs = [0.1, -1, 2, -3,              # root orient
                        -4, 5, -6,                   # root ang vel
                        17, -18, -19, 20, -21, 22, 23, -24, 25, -26,  # motor pos [1]
                         7,  -8,  -9, 10, -11, 12, -13, 14, -15, 16,  # motor pos [2]
                        37, -38, -39, 40, -41, 42, -43, 44, -45, 46,   # motor vel [1]
                        27, -28, -29, 30, -31, 32, -33, 34, -35, 36,   # motor vel [2]
        ]
        append_obs = [(len(base_mir_obs)+i) for i in range(3)] # 后面增加三个索引
        self.robot.clock_inds = append_obs[0:2]
        self.robot.mirrored_obs = np.array(base_mir_obs + append_obs, copy=True).tolist()
        self.robot.mirrored_acts = [-10, 11, -12, 13, -14, 15, -16, 17, -18, 19,
                                    0.1, -1, -2, 3, -4, 5, -6, 7, -8, 9,]

        # set action space
        action_space_size = len(self.robot.actuators)
        action = np.zeros(action_space_size)
        self.action_space = np.zeros(action_space_size)

        # set observation space
        self.base_obs_len = 50
        self.observation_space = np.zeros(self.base_obs_len)
        
        self.reset_model()

    def get_obs(self):
        # external state
        clock = [np.sin(2 * np.pi * self.task._phase / self.task._period),
                 np.cos(2 * np.pi * self.task._phase / self.task._period)]
        ext_state = np.concatenate((clock, [self.task._goal_speed_ref]))

        # internal state
        qpos = np.copy(self.interface.get_qpos())
        qvel = np.copy(self.interface.get_qvel())

        root_r, root_p = tf3.euler.quat2euler(qpos[3:7])[0:2]
        root_orient = tf3.euler.euler2quat(root_r, root_p, 0)
        root_ang_vel = qvel[3:6]

        motor_pos = self.interface.get_act_joint_positions()
        motor_vel = self.interface.get_act_joint_velocities()
        motor_pos = [motor_pos[i] for i in self.actuators]
        motor_vel = [motor_vel[i] for i in self.actuators]

        robot_state = np.concatenate([
            root_orient,
            root_ang_vel,
            motor_pos,
            motor_vel,
        ])
        state = np.concatenate([robot_state, ext_state])
        assert state.shape==(self.base_obs_len,)
        return state.flatten()

    def step(self, a):
        # 为什么先robot.step() -> 获取moter的action并让其产生动作
        #
        # make one control step
        applied_action = self.robot.step(a)

        # compute reward
        self.task.step()
        rewards = self.task.calc_reward(self.robot.prev_torque, self.robot.prev_action, applied_action)
        total_reward = sum([float(i) for i in rewards.values()])

        # check if terminate
        done = self.task.done()

        obs = self.get_obs()
        return obs, total_reward, done, rewards

    def reset_model(self):
        '''
        # dynamics randomization
        dofadr = [self.interface.get_jnt_qveladr_by_name(jn)
                  for jn in self.interface.get_actuated_joint_names()]
        for jnt in dofadr:
            self.model.dof_frictionloss[jnt] = np.random.uniform(0,10)    # actuated joint frictionloss
            self.model.dof_damping[jnt] = np.random.uniform(0.2,5)        # actuated joint damping
            self.model.dof_armature[jnt] *= np.random.uniform(0.90, 1.10) # actuated joint armature
        '''

        c = 0
        self.init_qpos = list(self.robot.init_qpos_)
        self.init_qvel = list(self.robot.init_qvel_)
        self.init_qpos = self.init_qpos + np.random.uniform(low=-c, high=c, size=self.model.nq)
        self.init_qvel = self.init_qvel + np.random.uniform(low=-c, high=c, size=self.model.nv)

        # modify init state acc to task
        root_adr = self.interface.get_jnt_qposadr_by_name('//unnamed_joint_0')[0]
        self.init_qpos[root_adr+2] = 0.1
        self.set_state(
            np.asarray(self.init_qpos),
            np.asarray(self.init_qvel)
        )
        obs = self.get_obs()
        self.task.reset()
        return obs

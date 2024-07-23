import os
import numpy as np
import transforms3d as tf3 # 3D空间中的几何变换
import collections

from tasks import stepping_task
from envs.common import mujoco_env
from envs.common import robot_interface
from envs.jvrc import robot

from .gen_xml import builder # 从当前包的gen_xml模块导入builder，可能用于生成或修改XML文件


class JvrcStepEnv(mujoco_env.MujocoEnv):
    def __init__(self):
        sim_dt = 0.0025
        control_dt = 0.025
        frame_skip = (control_dt/sim_dt)

        path_to_xml_out = '/tmp/mjcf-export/jvrc_step/jvrc1.xml'
        # 对scene.xml文件修改并导出
        if not os.path.exists(path_to_xml_out):
            builder(path_to_xml_out)
        mujoco_env.MujocoEnv.__init__(self, path_to_xml_out, sim_dt, control_dt)

        """
        funciton: 设置PD控制器的增益矩阵
        创建一个12x2的零矩阵;12行对应12个控制的关节（每条腿6个关节）;2列分别对应比例增益（P）和微分增益（D）。
        设置缩放系数，系数用于统一缩放所有增益值。
        设置比例增益和微分增益：
        1. 设置矩阵的第一列（转置后的第一行），对应比例增益。
        2. 每条腿的增益值模式：[200, 200, 200, 250, 80, 80]。
        3. 这些值分别对应髋部（3个自由度）、膝部和脚踝（2个自由度）的控制。
        另外：
        1. P增益（比例）通常较大，用于快速响应误差。
        2. D增益（微分）较小，用于抑制过冲和振荡。
        3. 使用coeff系数允许整体调整所有增益，而不改变它们之间的相对关系。
        4. 髋关节和膝关节的增益较高，可能是因为它们在保持平衡和控制步态中起着更重要的作用。
        5. 脚踝关节的增益较低，可能是为了允许更柔软的地面接触和适应性。
        """
        pdgains = np.zeros((12, 2))
        coeff = 0.5
        pdgains.T[0] = coeff * np.array([200, 200, 200, 250, 80, 80,
                                         200, 200, 200, 250, 80, 80,])
        pdgains.T[1] = coeff * np.array([20, 20, 20, 25, 8, 8,
                                         20, 20, 20, 25, 8, 8,])

        # list of desired actuators
        # RHIP_P, RHIP_R, RHIP_Y, RKNEE, RANKLE_R, RANKLE_P
        # LHIP_P, LHIP_R, LHIP_Y, LKNEE, LANKLE_R, LANKLE_P
        self.actuators = [0, 1, 2, 3, 4, 5,
                          6, 7, 8, 9, 10, 11]

        # set up interface
        # 创建一个RobotInterface实例，用于与仿真环境交互，参数包括模型、数据，以及右脚踝和左脚踝的标识符。
        self.interface = robot_interface.RobotInterface(self.model, self.data, 'R_ANKLE_P_S', 'L_ANKLE_P_S')

        # 创建一个SteppingTask实例，定义步行任务的参数。
        # 指定了控制时间步长、中立足部朝向（四元数表示）、根部（骨盆）、左右脚和头部的标识符。
        self.task = stepping_task.SteppingTask(client=self.interface,
                                               dt=control_dt,
                                               neutral_foot_orient=np.array([1, 0, 0, 0]),
                                               root_body='PELVIS_S',
                                               lfoot_body='L_ANKLE_P_S',
                                               rfoot_body='R_ANKLE_P_S',
                                               head_body='NECK_P_S',
        )
        # set goal height
        """
        设置目标高度为0.80米。
        设置总步态周期为1.1秒。
        设置摆动相持续时间为0.75秒。
        设置支撑相持续时间为0.35秒。
        """
        self.task._goal_height_ref = 0.80
        self.task._total_duration = 1.1
        self.task._swing_duration = 0.75
        self.task._stance_duration = 0.35

        # 创建一个JVRC机器人模型实例。
        # 传入PD增益矩阵、控制时间步长、执行器列表和接口。
        self.robot = robot.JVRC(pdgains.T, control_dt, self.actuators, self.interface)

        # 定义了观察和动作的镜像函数索引
        # 镜像观察值: 正值表示保持原值，负值表示取反，小数值（如0.1）可能表示特殊处理；包括根部朝向、角速度、电机位置和速度。
        base_mir_obs = [0.1, -1, 2, -3,              # root orient
                        -4, 5, -6,                   # root ang vel
                        13, -14, -15, 16, -17, 18,   # motor pos [1]
                         7,  -8,  -9, 10, -11, 12,   # motor pos [2]
                        25, -26, -27, 28, -29, 30,   # motor vel [1]
                        19, -20, -21, 22, -23, 24,   # motor vel [2]
        ]
        # 创建10个额外的观察索引
        append_obs = [(len(base_mir_obs)+i) for i in range(10)]
        self.robot.clock_inds = append_obs[0:2] # 时钟索引
        # 合并基础镜像观察和附加观察，创建完整的镜像观察列表
        self.robot.mirrored_obs = np.array(base_mir_obs + append_obs, copy=True).tolist()
        self.robot.mirrored_acts = [6, -7, -8, 9, -10, 11,
                                    0.1, -1, -2, 3, -4, 5,]

        # set action space
        action_space_size = len(self.robot.actuators)
        action = np.zeros(action_space_size)
        self.action_space = np.zeros(action_space_size)

        # set observation space
        self.base_obs_len = 41
        self.observation_space = np.zeros(self.base_obs_len)

    def get_obs(self):
        # 外部状态
        """
        创建一个周期性的时钟信号，用正弦和余弦表示当前相位。
        包含目标步骤的 x, y, z 坐标和 theta 角度。
        """
        clock = [np.sin(2 * np.pi * self.task._phase / self.task._period),
                 np.cos(2 * np.pi * self.task._phase / self.task._period)]
        ext_state = np.concatenate((clock,
                                    np.asarray(self.task._goal_steps_x).flatten(),
                                    np.asarray(self.task._goal_steps_y).flatten(),
                                    np.asarray(self.task._goal_steps_z).flatten(),
                                    np.asarray(self.task._goal_steps_theta).flatten()))

        # 内部状态
        # 获取位置和速度
        qpos = np.copy(self.interface.get_qpos())
        qvel = np.copy(self.interface.get_qvel())

        # 将根部朝向从四元数转换为欧拉角，但只取滚转角（roll）和俯仰角（pitch）。
        # 然后将这两个角度再转回四元数，忽略偏航角（yaw），设为0。
        # 提取根部角速度
        """
        【四元数的优点：避免万向节锁（gimbal lock）问题；计算效率高，特别是在连续旋转中；插值更加平滑。】
        四元数到欧拉角的转换:
        1. qpos[3:7] 代表机器人根部的朝向，以四元数形式表示。
        2. 四元数是一种表示3D旋转的方法，由四个分量组成：q = [w, x, y, z]。
        3. quat2euler 函数将四元数转换为欧拉角（roll, pitch, yaw）。
        4. [0:2] 只取前两个角度，即 roll (root_r) 和 pitch (root_p)。
        
        【欧拉角的优点：直观易懂，分别表示绕 x, y, z 轴的旋转；在某些应用中更容易可视化和理解。】
        欧拉角到四元数的转换：
        1. 将 roll 和 pitch 角度转回四元数，yaw 设为 0。
        2. 这步操作实际上是在忽略原始朝向中的 yaw 分量。
        
        角速度：提取根部的角速度，通常表示为绕 x, y, z 轴的旋转速度。
        
        其他：
        1. 为什么忽略 yaw
            1. 在许多机器人应用中，特别是步行机器人，yaw（偏航角）的绝对值往往不如 roll 和 pitch 重要。
            2. 忽略 yaw 可以简化问题，使机器人更关注保持平衡和前进方向。
        2. 角速度与四元数的关系
        """
        root_r, root_p = tf3.euler.quat2euler(qpos[3:7])[0:2]
        root_orient = tf3.euler.euler2quat(root_r, root_p, 0)
        root_ang_vel = qvel[3:6]

        # 获取所有关节的位置和速度。
        # 只选择由 self.actuators 指定的关节。
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

        # 确保状态向量的长度正确。
        # 返回展平的状态向量。
        assert state.shape==(self.base_obs_len,)
        return state.flatten()

    def step(self, a):
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
        # 动力学随机化
        dofadr = [self.interface.get_jnt_qveladr_by_name(jn)
                  for jn in self.interface.get_actuated_joint_names()]
        for jnt in dofadr:
            self.model.dof_frictionloss[jnt] = np.random.uniform(0,10)    # actuated joint frictionloss
            self.model.dof_damping[jnt] = np.random.uniform(0.2,5)        # actuated joint damping
            self.model.dof_armature[jnt] *= np.random.uniform(0.90, 1.10) # actuated joint armature
        '''

        """
        functions:对初始位置和速度添加小的随机扰动。
                  这有助于增加训练数据的多样性，提高学习的泛化能力。
        """
        c = 0.02
        self.init_qpos = list(self.robot.init_qpos_)
        self.init_qvel = list(self.robot.init_qvel_)
        self.init_qpos = self.init_qpos + np.random.uniform(low=-c, high=c, size=self.model.nq)
        self.init_qvel = self.init_qvel + np.random.uniform(low=-c, high=c, size=self.model.nv)

        """
        # 修改根节点的初始状态
        1. 特别设置根节点的位置和朝向。
        2. x 和 y 位置在 [-1, 1] 范围内随机。
        3. z 位置固定在 0.81（可能是机器人的标准高度）。
        4. 朝向使用欧拉角转四元数，pitch 在 ±5° 范围内，yaw 在 ±180° 范围内随机。
        """
        root_adr = self.interface.get_jnt_qposadr_by_name('root')[0]
        self.init_qpos[root_adr+0] = np.random.uniform(-1, 1)
        self.init_qpos[root_adr+1] = np.random.uniform(-1, 1)
        self.init_qpos[root_adr+2] = 0.81
        self.init_qpos[root_adr+3:root_adr+7] = tf3.euler.euler2quat(0, np.random.uniform(-5, 5)*np.pi/180, np.random.uniform(-np.pi, np.pi))
        self.set_state(
            self.init_qpos,
            self.init_qvel
        )
        self.task.reset(iter_count = self.robot.iteration_count)
        obs = self.get_obs()
        return obs

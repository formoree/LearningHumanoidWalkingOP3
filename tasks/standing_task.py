import numpy as np
import transforms3d as tf3
from tasks import rewards

class StandingTask(object):
    """Dynamically stable walking on biped."""

    def __init__(self,
                 client=None,
                 dt=0.025,
                 neutral_foot_orient=[],
                 root_body='pelvis',
                 lfoot_body='lfoot',
                 rfoot_body='rfoot',
                 waist_r_joint='waist_r',
                 waist_p_joint='waist_p',
                 head_body='head',
    ):

        self._client = client
        self._control_dt = dt
        self._neutral_foot_orient=neutral_foot_orient

        self._mass = self._client.get_robot_mass()

        # These depend on the robot, hardcoded for now
        # Ideally, they should be arguments to __init__
        self._goal_speed_ref = []
        self._goal_height_ref = []
        self._swing_duration = []
        self._stance_duration = []
        self._total_duration = []

        self._root_body_name = root_body
        self._lfoot_body_name = lfoot_body
        self._rfoot_body_name = rfoot_body
        self._head_body_name = head_body

    def calc_reward(self, prev_torque, prev_action, action):
        """
        初始化：计算目标方向的四元数；获取左右脚的力和速度时钟函数。
        定位：获取头部和根部（躯干）的 x-y 平面位置。
        reward构成：
        1. 足部力量奖励 (15%)
        2. 足部速度奖励 (15%)
        3. 身体方向奖励 (5%) 三个位置奖励/3
        4. 身体速度&加速度奖励 (5%)
        4. 高度误差奖励 (5%)
        5. 前向速度奖励 (20%)
        6. 上半身奖励 (5%)
        7。 扭矩奖励 (5%)
        """
        self.l_foot_vel = self._client.get_lfoot_body_vel()[0]
        self.r_foot_vel = self._client.get_rfoot_body_vel()[0]
        self.l_foot_frc = self._client.get_lfoot_grf()
        self.r_foot_frc = self._client.get_rfoot_grf()
        sim_dt = self._client.sim_dt()
        upstand_reward = rewards._calc_HeightStand_reward(self, sim_dt)
        quad_ctrl_cost = np.square(self._client.get_act_joint_torques()).sum()
        quad_impact_cost = np.square(self._client.get_body_ext_force()).sum()
        # print(upstand_reward, sim_dt)
        # print(quad_ctrl_cost)
        # print(quad_impact_cost)
        reward = dict(upstand_score = upstand_reward,
                      quad_ctrl_penalty = -0.01 * quad_ctrl_cost,
                      quad_impact_penalty = -min(0.1 * quad_impact_cost, 10)
        )
        return reward

    def step(self):
        # 实现一个周期性的计数器或相位跟踪器
        if self._phase>self._period:
            self._phase=0
        self._phase+=1
        return

    def done(self):
        """
        终止条件：
        qpos[2]_ll: 机器人的垂直位置（假设 qpos[2] 代表高度）低于 0.6 单位。
        qpos[2]_ul: 机器人的垂直位置高于 1.4 单位。
        contact_flag: 机器人发生自碰撞。
        """
        contact_flag = self._client.check_self_collisions()
        qpos = self._client.get_qpos()
        terminate_conditions = {"qpos[2]_ll":(qpos[2] < 0.01),
                                "qpos[2]_ul":(qpos[2] > 0.35),
                                "contact_flag":contact_flag,
        }

        done = True in terminate_conditions.values()
        return done

    def reset(self):
        # 随机选择目标速度，可能是 0 或 0.3 到 0.4 之间的随机值
        self._goal_speed_ref = 0
        # 为左右腿创建相位奖励，可能用于同步步态
        # self.right_clock, self.left_clock = rewards.create_phase_reward(self._swing_duration,
        #                                                                 self._stance_duration,
        #                                                                 0.1,
        #                                                                 "grounded",
        #                                                                 40)# 1/self._control_dt

        # number of control steps in one full cycle
        # (one full cycle includes left swing + right swing)
        self._period = np.floor(2*self._total_duration*(1/self._control_dt))
        # randomize phase during initialization
        self._phase = np.random.randint(0, self._period)

import sys
import os
from dm_control import mjcf
import mujoco
import mujoco_viewer
import random
import string

JVRC_DESCRIPTION_PATH="models/robotis_op3/scene.xml"

def builder(export_path):

    print("Modifying XML model...")
    mjcf_model = mjcf.from_path(JVRC_DESCRIPTION_PATH)

    # # # set njmax and nconmax
    # """
    # 设置 njmax 和 nconmax 为 -1，这可能意味着使用默认值或无限制。
    # """
    mjcf_model.size.njmax = -1
    mjcf_model.size.nconmax = -1


    # modify skybox
    for tx in mjcf_model.asset.texture:
        if tx.type=="skybox":
            tx.rgb1 = '1 1 1'
            tx.rgb2 = '1 1 1'

    # 移除所有碰撞，并且定义一系列关节组
    mjcf_model.contact.remove()

    head_joints = ['head_pan', 'head_tilt']
    arm_joints = ['l_sho_pitch', 'l_sho_roll', 'l_el',
                   'r_sho_pitch', 'r_sho_roll', 'r_el']
    leg_joints = ['r_hip_yaw', 'r_hip_roll', 'r_hip_pitch', 'r_knee', 'r_ank_pitch', 'r_ank_roll',
                  'l_hip_yaw', 'l_hip_roll', 'l_hip_pitch', 'l_knee', 'l_ank_pitch', 'l_ank_roll']

    # # 只保留腿部关节执行器
    for mot in mjcf_model.actuator.position:
        if mot.joint.name not in leg_joints:
            mot.remove()

    # # remove unused joints
    # # 除了腿部关节，其他都被移除，因为我们只进行行走
    for joint in  head_joints:
        mjcf_model.find('joint', joint).remove()

    # remove existing equality -> 约束
    mjcf_model.equality.remove()

    # add equality for arm joints
    """
    polycoef 多项式系数
    1. 通过添加这些约束，手臂关节被设置到特定的位置。
    2. 多项式系数的第一个值（例如 -0.052, -0.169 等）可能表示关节的目标角度或偏移量
    3. 这些约束可能用于模拟手臂的被动动力学，即手臂不主动移动，但会根据身体其他部分的运动而被动摆动
    4. 通过固定手臂位置，控制问题被简化，研究者可以专注于腿部动作的控制。
    5. 使用多项式系数允许更精确地控制关节位置，而不仅仅是简单的固定位置。
    """
    arm_joints = ['l_sho_pitch', 'l_sho_roll', 'l_el',
                  'r_sho_pitch', 'r_sho_roll', 'r_el']
    mjcf_model.equality.add('joint', joint1=arm_joints[0], polycoef='-0.0314 0 0 0 0')
    mjcf_model.equality.add('joint', joint1=arm_joints[1], polycoef='1.23 0 0 0 0')
    mjcf_model.equality.add('joint', joint1=arm_joints[2], polycoef='-0.44 0 0 0 0')
    mjcf_model.equality.add('joint', joint1=arm_joints[3], polycoef='-0.0314 0 0 0 0')
    mjcf_model.equality.add('joint', joint1=arm_joints[4], polycoef='-1.23 0 0 0 0')
    mjcf_model.equality.add('joint', joint1=arm_joints[5], polycoef='0.44 0 0 0 0')

    # collision geoms
    # 定义需要保留碰撞检测的身体部位的列表，主要是腿部的关键关节
    collision_geoms = [
        'l_hip_roll_link', 'l_hip_yaw_link', 'l_knee_link',
        'r_hip_roll_link', 'r_hip_yaw_link', 'r_knee_link',
    ]

    # remove unused collision geoms
    for body in mjcf_model.worldbody.find_all('body'):
        for idx, geom in enumerate(body.geom):
            geom.name = body.name + '-geom-' + repr(idx)
            if (geom.dclass.dclass=="collision"):
                if body.name not in collision_geoms:
                    geom.remove()

    # move collision geoms to different group
    mjcf_model.default.default['collision'].geom.group = 3

    # manually create collision geom for feet
    """
    为右脚和左脚分别添加了一个盒状的碰撞几何体
    这些几何体被设置为特定的尺寸和位置，可能是为了更准确地模拟脚部与地面的接触
    """
    # mjcf_model.worldbody.find('body', 'r_ank_pitch_link').add('geom', dclass='collision', size='0.1 0.05 0.01', pos='-0.0241 0.019 0', type='box')
    # mjcf_model.worldbody.find('body', 'l_ank_pitch_link').add('geom', dclass='collision', size='0.1 0.05 0.01', pos='-0.0241 0.019 0', type='box')

    # ignore collision
    # """
    # 这里排除了膝部和脚踝部位之间的碰撞检测。
    # 这可能是为了避免不必要的自碰撞检测，提高仿真效率。
    # """
    mjcf_model.contact.add('exclude', body1='r_knee_link', body2='r_ank_pitch_link')
    mjcf_model.contact.add('exclude', body1='l_knee_link', body2='l_ank_pitch_link')
    #
    # mjcf_model.find('geom', 'floor').remove()
    # mjcf_model.worldbody.add('body', name='floor')
    # mjcf_model.find('body', 'floor').add('geom', name='floor', type="plane", size="0 0 0.25", material="groundplane")

    # export model
    mjcf.export_with_assets(mjcf_model, out_dir=os.path.dirname(export_path), out_file_name=export_path, precision=5)
    print("Exporting XML model to ", export_path)
    return

if __name__=='__main__':
    builder(sys.argv[1])

import sys
import os
from dm_control import mjcf
import random
import string

JVRC_DESCRIPTION_PATH="models/jvrc_mj_description/xml/scene.xml"

def builder(export_path):

    print("Modifying XML model...")
    mjcf_model = mjcf.from_path(JVRC_DESCRIPTION_PATH)

    # set njmax and nconmax
    """
    设置 njmax 和 nconmax 为 -1，这可能意味着使用默认值或无限制。
    设置平均尺寸 meansize 为 0.1。
    设置平均质量 meanmass 为 2。
    """
    mjcf_model.size.njmax = -1
    mjcf_model.size.nconmax = -1
    mjcf_model.statistic.meansize = 0.1
    mjcf_model.statistic.meanmass = 2

    # modify skybox
    for tx in mjcf_model.asset.texture:
        if tx.type=="skybox":
            tx.rgb1 = '1 1 1'
            tx.rgb2 = '1 1 1'

    # 移除所有碰撞，并且定义一系列关节组
    mjcf_model.contact.remove()

    waist_joints = ['WAIST_Y', 'WAIST_P', 'WAIST_R']
    head_joints = ['NECK_Y', 'NECK_R', 'NECK_P']
    hand_joints = ['R_UTHUMB', 'R_LTHUMB', 'R_UINDEX', 'R_LINDEX', 'R_ULITTLE', 'R_LLITTLE',
                   'L_UTHUMB', 'L_LTHUMB', 'L_UINDEX', 'L_LINDEX', 'L_ULITTLE', 'L_LLITTLE']
    arm_joints = ['R_SHOULDER_Y', 'R_ELBOW_Y', 'R_WRIST_R', 'R_WRIST_Y',
                  'L_SHOULDER_Y', 'L_ELBOW_Y', 'L_WRIST_R', 'L_WRIST_Y']
    leg_joints = ['R_HIP_P', 'R_HIP_R', 'R_HIP_Y', 'R_KNEE', 'R_ANKLE_R', 'R_ANKLE_P',
                  'L_HIP_P', 'L_HIP_R', 'L_HIP_Y', 'L_KNEE', 'L_ANKLE_R', 'L_ANKLE_P']

    # remove actuators except for leg joints
    # 只保留腿部关节执行器
    for mot in mjcf_model.actuator.motor:
        if mot.joint.name not in leg_joints:
            mot.remove()

    # remove unused joints
    # 除了腿部关节，其他都被移除，因为我们只进行行走
    for joint in waist_joints + head_joints + hand_joints + arm_joints:
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
    arm_joints = ['R_SHOULDER_P', 'R_SHOULDER_R', 'R_ELBOW_P',
                  'L_SHOULDER_P', 'L_SHOULDER_R', 'L_ELBOW_P']
    mjcf_model.equality.add('joint', joint1=arm_joints[0], polycoef='-0.052 0 0 0 0')
    mjcf_model.equality.add('joint', joint1=arm_joints[1], polycoef='-0.169 0 0 0 0')
    mjcf_model.equality.add('joint', joint1=arm_joints[2], polycoef='-0.523 0 0 0 0')
    mjcf_model.equality.add('joint', joint1=arm_joints[3], polycoef='-0.052 0 0 0 0')
    mjcf_model.equality.add('joint', joint1=arm_joints[4], polycoef='0.169 0 0 0 0')
    mjcf_model.equality.add('joint', joint1=arm_joints[5], polycoef='-0.523 0 0 0 0')

    # collision geoms
    # 定义需要保留碰撞检测的身体部位的列表，主要是腿部的关键关节
    collision_geoms = [
        'R_HIP_R_S', 'R_HIP_Y_S', 'R_KNEE_S',
        'L_HIP_R_S', 'L_HIP_Y_S', 'L_KNEE_S',
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
    mjcf_model.worldbody.find('body', 'R_ANKLE_P_S').add('geom', dclass='collision', size='0.1 0.05 0.01', pos='0.02373 -0.01037 -0.0276', type='box')
    mjcf_model.worldbody.find('body', 'L_ANKLE_P_S').add('geom', dclass='collision', size='0.1 0.05 0.01', pos='0.02373 -0.01037 -0.0276', type='box')

    # ignore collision
    """
    这里排除了膝部和脚踝部位之间的碰撞检测。
    这可能是为了避免不必要的自碰撞检测，提高仿真效率。
    """
    mjcf_model.contact.add('exclude', body1='R_KNEE_S', body2='R_ANKLE_P_S')
    mjcf_model.contact.add('exclude', body1='L_KNEE_S', body2='L_ANKLE_P_S')

    # remove unused meshes
    meshes = [g.mesh.name for g in mjcf_model.find_all('geom') if g.type=='mesh' or g.type==None]
    for mesh in mjcf_model.find_all('mesh'):
        if mesh.name not in meshes:
            mesh.remove()

    # fix site pos
    # 调整了右脚和左脚力传感器的位置
    mjcf_model.worldbody.find('site', 'rf_force').pos = '0.03 0.0 -0.1'
    mjcf_model.worldbody.find('site', 'lf_force').pos = '0.03 0.0 -0.1'

    # add box geoms
    """
    这段代码添加了20个盒子几何体。
    每个盒子都被添加为一个独立的刚体（body），位置在z=-0.2。
    每个盒子的几何体被设置为碰撞类型（collision），属于组0，大小为1x1x0.1的立方体。
    """
    for idx in range(20):
        name = 'box' + repr(idx+1).zfill(2)
        mjcf_model.worldbody.add('body', name=name, pos=[0, 0, -0.2])
        mjcf_model.find('body', name).add('geom',
                                          name=name,
                                          dclass='collision',
                                          group='0',
                                          size='1 1 0.1',
                                          type='box',
                                          material='')

    # wrap floor geom in a body
    """
    首先移除原有的地板几何体。
    然后添加一个新的刚体作为地板。
    在这个新的刚体上添加一个平面几何体作为新的地板，设置其材质为"groundplane"。
    """
    mjcf_model.find('geom', 'floor').remove()
    mjcf_model.worldbody.add('body', name='floor')
    mjcf_model.find('body', 'floor').add('geom', name='floor', type="plane", size="0 0 0.25", material="groundplane")

    # export model
    mjcf.export_with_assets(mjcf_model, out_dir=os.path.dirname(export_path), out_file_name=export_path, precision=5)
    print("Exporting XML model to ", export_path)
    return

if __name__=='__main__':
    builder(sys.argv[1])

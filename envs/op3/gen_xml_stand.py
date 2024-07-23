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

    # modify skybox
    for tx in mjcf_model.asset.texture:
        if tx.type=="skybox":
            tx.rgb1 = '1 1 1'
            tx.rgb2 = '1 1 1'

    mjcf_model.find('joint', 'head_pan').range = [-0.79, 0.79]
    mjcf_model.find('joint', 'head_tilt').range = [-0.63, -0.16]
    mjcf_model.find('joint', 'l_ank_pitch').range = [-0.4, 1.8]
    mjcf_model.find('joint', 'l_ank_roll').range = [-0.4, 0.4]
    mjcf_model.find('joint', 'l_el').range = [-1.4, 0.2]
    mjcf_model.find('joint', 'l_hip_pitch').range = [-1.6, 0.5]
    mjcf_model.find('joint', 'l_hip_roll').range = [-0.4, -0.1]
    mjcf_model.find('joint', 'l_hip_yaw').range = [-0.3, 0.3]
    mjcf_model.find('joint', 'l_knee').range = [-0.2, 2.2]
    mjcf_model.find('joint', 'l_sho_pitch').range = [-2.2, 2.2]
    mjcf_model.find('joint', 'l_sho_roll').range = [-0.8, 1.6]
    mjcf_model.find('joint', 'r_ank_pitch').range = [-1.8, 0.4]
    mjcf_model.find('joint', 'r_ank_roll').range = [-0.4, 0.4]
    mjcf_model.find('joint', 'r_el').range = [-0.2, 1.4]
    mjcf_model.find('joint', 'r_hip_pitch').range = [-0.5, 1.6]
    mjcf_model.find('joint', 'r_hip_roll').range = [0.1, 0.4]
    mjcf_model.find('joint', 'r_hip_yaw').range = [-0.3, 0.3]
    mjcf_model.find('joint', 'r_knee').range = [-2.2, 0.2]
    mjcf_model.find('joint', 'r_sho_pitch').range = [-2.2, 2.2]
    mjcf_model.find('joint', 'r_sho_roll').range = [-1.6, 0.8]

    # # collision geoms
    # # 定义需要保留碰撞检测的身体部位的列表，主要是腿部的关键关节
    # collision_geoms = [
    #     'l_hip_roll_link', 'l_hip_yaw_link', 'l_knee_link',
    #     'r_hip_roll_link', 'r_hip_yaw_link', 'r_knee_link',
    # ]
    #
    # # remove unused collision geoms
    # for body in mjcf_model.worldbody.find_all('body'):
    #     for idx, geom in enumerate(body.geom):
    #         geom.name = body.name + '-geom-' + repr(idx)
    #         if (geom.dclass.dclass=="collision"):
    #             if body.name not in collision_geoms:
    #                 geom.remove()
    #
    # # move collision geoms to different group
    # mjcf_model.default.default['collision'].geom.group = 3
    #
    # # ignore collision
    # """
    # 这里排除了膝部和脚踝部位之间的碰撞检测。
    # 这可能是为了避免不必要的自碰撞检测，提高仿真效率。
    # """
    # mjcf_model.contact.add('exclude', body1='r_knee_link', body2='r_ank_pitch_link')
    # mjcf_model.contact.add('exclude', body1='l_knee_link', body2='l_ank_pitch_link')
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

# LearningHumanoidWalking

Original Code: [rohanpsingh/LearningHumanoidWalking: Training a humanoid robot for locomotion using Reinforcement Learning (github.com)](https://github.com/rohanpsingh/LearningHumanoidWalking)

I have changed the robot model to robot OP3, and adjust the parameters and related files.

1. If your question pertains to converting URDF files into XML format, consider searching for tutorials related to MuJoCo.
2. Once you have generated the XML file for the OP3 robot, you can modify your model using **`envs/jvrc/gen_xml.py`**. This script is specifically designed to transform a complete robot model into a version that is more conducive to training.
3. Next, you may need to refer to **`envs/jvrc/jvrc_walk.py`** in order to create your own **`op3_walk.py`** file. In this file, it will be necessary to adjust the PD gains and mirror symmetry settings. Additionally, ensure that you replace the XML path with your own.
4. Another file that may require adjustments is **`envs/jvrc/robot.py`**, as the half_sitting_pose used for the JVRC might not be appropriate for your robot. When training a robot to walk, it’s optimal for its initial knee position to be slightly bent to facilitate movement.

## Code structure:
A rough outline for the repository that might be useful for adding your own robot:
```
LearningHumanoidWalking/
├── envs/                <-- Actions and observation space, PD gains, simulation step, control decimation, init, ...
├── tasks/               <-- Reward function, termination conditions, and more...
├── rl/                  <-- Code for PPO, actor/critic networks, observation normalization process...
├── models/              <-- MuJoCo model files: XMLs/meshes/textures
├── trained/             <-- Contains pretrained model for JVRC
└── scripts/             <-- Utility scripts, etc.
```

## Requirements:
- Python version: 3.7.11  
- [Pytorch](https://pytorch.org/)
- pip install:
  - mujoco==2.2.0
  - [mujoco-python-viewer](https://github.com/rohanpsingh/mujoco-python-viewer)
  - ray==1.9.2
  - transforms3d
  - matplotlib
  - scipy

## Usage:

Environment names supported:  

| Task Description      | Environment name |
| ----------- | ----------- |
| Basic Walking Task   | 'jvrc_walk' |
| Stepping Task (using footsteps)  | 'jvrc_step' |
| Walking Task for OP3 | 'OP3_walk' |

#### **To train:**

```
$ python run_experiment.py train --logdir <path_to_exp_dir> --num_procs <num_of_cpu_procs> --env <name_of_environment>
```


#### **To play:**

We need to write a script specific to each environment.    
For example, `debug_stepper.py` can be used with the `jvrc_step` environment.  
```
$ PYTHONPATH=.:$PYTHONPATH python scripts/debug_stepper.py --path <path_to_exp_dir>
```

#### **What you should see:**

*Ascending stairs:*  
![climb_up](https://user-images.githubusercontent.com/16384313/180697513-25796b1a-87e0-4ab2-9e5f-d86c58ebea36.gif)

*Descending stairs:*  
![climb_down](https://user-images.githubusercontent.com/16384313/180697788-d1a2eec0-0d3d-451a-95e0-9f0e60191c34.gif)

*Walking on curves:*  
![curve](https://user-images.githubusercontent.com/16384313/180697266-7b44beb3-38bf-4494-b568-963919dc1106.gif)


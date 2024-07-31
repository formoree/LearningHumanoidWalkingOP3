import os
import sys # 与python解释器交互
import argparse
import ray # 用于并行和分布式计算
from functools import partial # 函数式编程函数

import numpy as np
import torch
import pickle # 用于序列化和反序列化

from rl.algos.ppo import PPO
from rl.policies.actor import Gaussian_FF_Actor
from rl.policies.critic import FF_V
from rl.envs.normalize import get_normalization_params
from rl.envs.wrappers import SymmetricEnv

def import_env(env_name_str):
    if env_name_str=='jvrc_walk':
        from envs.jvrc import JvrcWalkEnv as Env
    elif env_name_str=='jvrc_step':
        from envs.jvrc import JvrcStepEnv as Env
    elif env_name_str=='op3_walk':
        from envs.op3 import Op3WalkEnv as Env
    elif env_name_str=='op3_stand':
        from envs.op3 import Op3StandEnv as Env
    else:
        raise Exception("Check env name!")
    return Env

def run_experiment(args):
    # import the correct environment
    Env = import_env(args.env)

    # wrapper function for creating parallelized envs
    env_fn = partial(Env) # 传递一个无参数的函数来创建环境
    if not args.no_mirror:
        try:
            print("Wrapping in SymmetricEnv.")
            """
            Env -> env_fn -> env_fn [partial]
            新创建的 env_fn 是 SymmetricEnv 的部分应用
            """
            env_fn = partial(SymmetricEnv, env_fn,
                             mirrored_obs=env_fn().robot.mirrored_obs,
                             mirrored_act=env_fn().robot.mirrored_acts,
                             clock_inds=env_fn().robot.clock_inds)
        except AttributeError as e:
            print("Warning! Cannot use SymmetricEnv.", e)
    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]

    # Set up Parallelism
    os.environ['OMP_NUM_THREADS'] = '1' # 设置环境变量 OMP -> OpenMP 并行编程API
    if not ray.is_initialized():
        ray.init(num_cpus=args.num_procs) #避免重复初始化Ray【分布式计算框架】

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 根据继续训练与否 选择模型actor and critic
    if args.continued:
        path_to_actor = ""
        path_to_pkl = ""
        if os.path.isfile(args.continued) and args.continued.endswith(".pt"):
            path_to_actor = args.continued
        if os.path.isdir(args.continued):
            path_to_actor = os.path.join(args.continued, "actor.pt")
        path_to_critic = path_to_actor.split('actor')[0]+'critic'+path_to_actor.split('actor')[1]
        policy = torch.load(path_to_actor)
        critic = torch.load(path_to_critic)
    else:
        policy = Gaussian_FF_Actor(obs_dim, action_dim, fixed_std=np.exp(args.std_dev), bounded=False)
        critic = FF_V(obs_dim)
        """
        function: 计算和设置观察空间（observation space）的归一化参数
        1. 计算得到的归一化参数（均值和标准差）赋值给 policy 对象的属性
        2. map(torch.Tensor, ...) 将结果转换为 PyTorch 张量
        3. get_normalization_params()
        """
        with torch.no_grad():
            policy.obs_mean, policy.obs_std = map(torch.Tensor,
                                                  get_normalization_params(iter=args.input_norm_steps,
                                                                           noise_std=1,
                                                                           policy=policy,
                                                                           env_fn=env_fn,
                                                                           procs=args.num_procs))
        critic.obs_mean = policy.obs_mean
        critic.obs_std = policy.obs_std
    
    policy.train()
    critic.train()

    # dump hyperparameters
    os.makedirs(args.logdir, exist_ok=True)
    pkl_path = os.path.join(args.logdir, "experiment.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(args, f)

    algo = PPO(args=vars(args), save_path=args.logdir)
    algo.train(env_fn, policy, critic, args.n_itr, anneal_rate=args.anneal)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # 检测arg是训练阶段么，并且在传入模型时进行删除
    # if sys.argv[1] != 'train':
    #     raise Exception("Invalid usage.")
    #
    # sys.argv.remove(sys.argv[1])
    parser.add_argument("--env", default="op3_walk", type=str)                        # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=0, type=int)                        # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--logdir", type=str, default="./logs_dir/")          # Where to log diagnostics to
    parser.add_argument("--input_norm_steps", type=int, default=100000) # 100000
    parser.add_argument("--n_itr", type=int, default=20000, help="Number of iterations of the learning algorithm")
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate") # 1e-4
    parser.add_argument("--eps", type=float, default=1e-5, help="Adam epsilon (for numerical stability)")
    parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
    parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
    parser.add_argument("--anneal", default=1.0, action='store_true', help="anneal rate for stddev")
    parser.add_argument("--std_dev", type=int, default=-1.5, help="exponent of exploration std_dev")
    parser.add_argument("--entropy_coeff", type=float, default=0.0, help="Coefficient for entropy regularization")
    parser.add_argument("--clip", type=float, default=0.2, help="Clipping parameter for PPO surrogate loss")
    parser.add_argument("--minibatch_size", type=int, default=64, help="Batch size for PPO updates")
    parser.add_argument("--epochs", type=int, default=3, help="Number of optimization epochs per PPO update") #Xie
    parser.add_argument("--use_gae", type=bool, default=True,help="Whether or not to calculate returns using Generalized Advantage Estimation")
    parser.add_argument("--num_procs", type=int, default=1, help="Number of threads to train on")
    parser.add_argument("--max_grad_norm", type=float, default=0.05, help="Value to clip gradients at.")
    parser.add_argument("--max_traj_len", type=int, default=400, help="Max episode horizon")
    parser.add_argument("--no_mirror", required=False, action="store_true", help="to use SymmetricEnv")
    parser.add_argument("--mirror_coeff", required=False, default=0.4, type=float, help="weight for mirror loss")
    parser.add_argument("--eval_freq", required=False, default=400, type=int, help="Frequency of performing evaluation")
    parser.add_argument("--continued", required=False, default=None, type=str, help="path to pretrained weights")
    args = parser.parse_args()

    run_experiment(args)

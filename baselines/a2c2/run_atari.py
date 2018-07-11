#!/usr/bin/env python3

from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c2.a2c import learn
from baselines.a2c2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, CapsulePolicy

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env):
    nstates = None
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
        nstates = 512
    elif policy == 'caps':
        policy_fn = CapsulePolicy
    env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
    learn(policy_fn, env, seed, nsteps=5, nstates=nstates, total_timesteps=int(num_timesteps * 1.1),  sc_coef=None, lrschedule=lrschedule)
    env.close()

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'caps'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_env=16)

if __name__ == '__main__':
    main()

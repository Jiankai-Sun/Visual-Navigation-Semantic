from __future__ import division
import os
# os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch

from environment import atari_env

from utils import setup_logger, normalize_rgb_obs
from model import A3Clstm
from player_util import Agent
import logging
from gym.configuration import undo_logger_setup
import numpy as np
import cv2

undo_logger_setup()

parser = argparse.ArgumentParser(description='A3C_EVAL')
parser.add_argument(
    '--env',
    default='Default',
    metavar='ENV',
    help='environment to train on (default: Default)')
parser.add_argument(
    '--num-episodes',
    type=int,
    default=100,
    metavar='NE',
    help='how many episodes in evaluation (default: 100)')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--log-dir', 
    default='logs/', 
    metavar='LG', 
    help='folder to save logs')
parser.add_argument(
    '--render',
    default=False,
    metavar='R',
    help='Watch game as it being played')
parser.add_argument(
    '--render-freq',
    type=int,
    default=1,
    metavar='RF',
    help='Frequency to watch rendered game play')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=100,
    metavar='M',
    help='maximum length of an episode (default: 100)')
parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    help='GPU to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--hardness',
    type=float,
    default=1,
    metavar='HA',
    help='hardness of game (default: 1)')

args = parser.parse_args()

gpu_id = args.gpu_ids

torch.manual_seed(args.seed)
if gpu_id >= 0:
    torch.cuda.manual_seed(args.seed)

saved_state = torch.load(
    '{0}{1}.dat'.format(args.load_model_dir, args.env),
    map_location=lambda storage, loc: storage)

log = {}
setup_logger('{}_mon_log'.format(args.env), r'{0}{1}_mon_log'.format(
    args.log_dir, args.env))
log['{}_mon_log'.format(args.env)] = logging.getLogger(
    '{}_mon_log'.format(args.env))

d_args = vars(args)
for k in d_args.keys():
    log['{}_mon_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

env = atari_env(env_id=0, args=args, type='train')
num_tests = 0
reward_total_sum = 0
player = Agent(None, env, args, None)
player.model = A3Clstm(player.env.observation_space.shape[2],
                       player.env.action_space.n)

player.model.load_state_dict(saved_state)

if gpu_id >= 0:
    with torch.cuda.device(gpu_id):
        player.model = player.model.cuda()

# player.env = gym.wrappers.Monitor(
#     player.env, "{}_monitor".format(args.env), force=True)
player.model.eval()

for i_episode in range(args.num_episodes):
    player.state = player.env.reset()
    player.state = normalize_rgb_obs(player.state)
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
    reward_sum = 0
    while True:
        player.action_test()
        reward_sum += player.reward

        mat = np.array(player.env.debug_show())
        # mat = np.array(cv2.resize(player.env.debug_show(), (480, 360)))
        cv2.putText(mat, "action:"+str(player.action)+"reward:"+str(player.reward)+
                    " target:"+str(player.env.info['target_room']), (15, 15),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Visual Navigation", mat)
        cv2.moveWindow("Visual Navigation", 0, 0)
        cv2.waitKey(200)


        if player.done:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_mon_log'.format(args.env)].info(
                "episode length: {0}, reward sum: {1}, reward mean: {2:.4f}".format(
                    player.eps_len, reward_sum, reward_mean))
            player.eps_len = 0

            cv2.putText(mat, 'Terminated', (160, 200), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Visual Navigation", mat)
            cv2.moveWindow("Visual Navigation", 0, 0)
            cv2.waitKey(400)
            cv2.destroyAllWindows()


            break

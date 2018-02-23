from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from environment import atari_env
from utils import setup_logger, normalize_rgb_obs, frame_to_video
from model import A3Clstm
from player_util import Agent
import time
import logging
from tensorboardX import SummaryWriter
import cv2, os, shutil

def test(rank, args, shared_model):
    ptitle('Test Agent')
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    writer = SummaryWriter(log_dir=args.log_dir+'tb_test')
    log = {}
    setup_logger('{}_log'.format('Test_'+str(rank)),
                 r'{0}{1}_log'.format(args.log_dir, 'Test_'+str(rank)))
    log['{}_log'.format('Test_'+str(rank))] = logging.getLogger(
        '{}_log'.format('Test_'+str(rank)))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format('Test_'+str(rank))].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
    env = atari_env(env_id=rank, args=args, type='train')
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    num_inside_target_room = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Clstm(
        player.env.observation_space.shape[2], player.env.action_space.n)

    player.state = player.env.reset()
    player.state = normalize_rgb_obs(player.state)
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()

    player.model.eval()

    action_times = 0
    while True:
        action_times += 1
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())

        player.action_test()
        reward_sum += player.reward

        if not os.path.exists(args.log_dir + "video/" + str(rank) + "_" + str(num_tests)):
            os.makedirs(args.log_dir + "video/" + str(rank) + "_" + str(num_tests))

        cv2.imwrite(args.log_dir + "video/" + str(rank) + "_" + str(num_tests) + "/" + str(action_times) + ".png",
                    player.env.get_rgb())  # (90, 120, 3)

        if player.done:
            frame_to_video(fileloc=args.log_dir + "video/" + str(rank) + "_" + str(num_tests) + "/%d.png", t_w=120, t_h=90,
                           destination=args.log_dir + "video/" + str(rank) + "_" + str(num_tests) + ".mp4")
            shutil.rmtree(args.log_dir + "video/" + str(rank) + "_" + str(num_tests))
            action_times = 0
            num_tests += 1
            num_inside_target_room += player.env.inside_target_room
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            success_rate = num_inside_target_room / num_tests
            log['{}_log'.format('Test_'+str(rank))].info(
                "Time {0}, Tester {1}, test counter {2}, episode reward {3}, episode length {4}, reward mean {5:.4f}, success rate {6}".
                    format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)), rank,
                    num_tests, reward_sum, player.eps_len, reward_mean, success_rate))
            # Tensorboard
            writer.add_scalar("data/episode_reward", reward_sum, num_tests)
            writer.add_scalar("data/episode_length", player.eps_len, num_tests)
            writer.add_scalar("data/reward_mean", reward_mean, num_tests)
            writer.add_scalar("data/success_rate", success_rate, num_tests)

            if reward_sum > args.save_score_level:
                # player.model.load_state_dict(shared_model.state_dict())
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}_{2}.dat'.format(
                            args.save_model_dir, 'Test_' + str(rank), reward_sum))
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, '{0}{1}_{2}.dat'.format(
                        args.save_model_dir, 'Test_'+str(rank), reward_sum))

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            time.sleep(10)
            state = normalize_rgb_obs(state)
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()


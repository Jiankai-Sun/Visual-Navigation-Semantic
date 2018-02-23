from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads, normalize_rgb_obs
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
from utils import setup_logger
import logging, os, cv2
from tensorboardX import SummaryWriter
from utils import save_image
from constants import DEPTH_LOSS_DISCOUNT

def train(rank, args, shared_model, optimizer):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]

    writer = SummaryWriter(log_dir=args.log_dir+'tb_train')
    log = {}
    setup_logger('{}_train_log'.format(rank),
                 r'{0}{1}_train_log'.format(args.log_dir, rank))
    log['{}_train_log'.format(rank)] = logging.getLogger(
        '{}_train_log'.format(rank))
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    env = atari_env(env_id=rank, args=args, type='train')
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id

    player.model = A3Clstm(
        player.env.observation_space.shape[2], player.env.action_space.n)

    player.state = player.env.reset()
    player.state = normalize_rgb_obs(player.state)
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()
    player.model.train()
    num_trains = 0

    if not os.path.exists(args.log_dir + "images/"):
        os.makedirs(args.log_dir + "images/")

    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())
        for step in range(args.num_steps):
            player.action_train()

            if player.done:
                break

        if player.done:
            num_trains += 1
            log['{}_train_log'.format(rank)].info('entropy:{0}'.format(player.entropy.data[0]))
            writer.add_scalar("data/entropy_" + str(rank), player.entropy.data[0], num_trains)
            writer.add_image('FCN_' + str(rank), player.fcn, num_trains)
            writer.add_image('Depth_GroundTruth_' + str(rank), player.depth, num_trains)
            writer.add_image('RGB_' + str(rank), player.env.get_rgb(), num_trains)

            save_image(player.fcn.data, args.log_dir + "images/" + str(rank)+"_"+str(num_trains)+"_fcn.png")
            # print("player.fcn.data:", player.fcn.data)
            save_image(player.depth.data, args.log_dir + "images/" + str(rank) + "_" + str(num_trains) + "_depth.png")
            cv2.imwrite(args.log_dir + "images/" + str(rank) + "_" + str(num_trains) + "_rgb.png",
                        player.env.get_rgb())
            # print("player.depth.data:", player.depth.data)

            player.eps_len = 0
            player.current_life = 0
            state = player.env.reset()
            state = normalize_rgb_obs(state)
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        R = torch.zeros(1, 1)
        if not player.done:
            with torch.cuda.device(gpu_id):
                value, _, _, _ = player.model(
                    (Variable(player.state.unsqueeze(0)), (player.hx, player.cx),
                     Variable(torch.from_numpy(player.env.target).type(torch.FloatTensor).cuda())))
                R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = args.gamma * player.values[i + 1].data + player.rewards[i] - player.values[i].data

            gae = gae * args.gamma * args.tau + delta_t

            # policy_loss =  policy_loss - \
            #     player.log_probs[i] * \
            #     Variable(gae) - 0.01 * player.entropies[i] \
            #     + player.fcn_losses[i] # FCN

            policy_loss =  policy_loss - 1e-5*(player.log_probs[i] * Variable(gae)) - 1e-5*(0.01 * player.entropies[i]) \
                + player.fcn_losses[i] * DEPTH_LOSS_DISCOUNT # FCN

            # policy_loss = policy_loss + player.fcn_losses[i]  # FCN

        writer.add_scalar("data/value_loss_" + str(rank), value_loss, num_trains)
        writer.add_scalar("data/policy_loss_" + str(rank), policy_loss, num_trains)

        player.model.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(player.model.parameters(), 40.0)
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        player.clear_actions()


from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import normalize_rgb_obs#, CrossEntropyLoss2d, cross_entropy2d, normalize_depth


class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        self.max_length = False
        self.entropy = 0
        self.fcn_losses = []
        self.depth = None
        self.target = None

    def action_train(self):
        if self.done:
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    self.cx = Variable(torch.zeros(1, 512).cuda())
                    self.hx = Variable(torch.zeros(1, 512).cuda())
                    self.target = Variable(torch.from_numpy(self.env.target).type(torch.FloatTensor).cuda())
            else:
                self.cx = Variable(torch.zeros(1, 512))
                self.hx = Variable(torch.zeros(1, 512))
                self.target = Variable(torch.from_numpy(self.env.target).type(torch.FloatTensor))
        else:
            self.cx = Variable(self.cx.data)
            self.hx = Variable(self.hx.data)
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    self.target = Variable(torch.from_numpy(self.env.target).type(torch.FloatTensor).cuda())
            else:
                self.target = Variable(torch.from_numpy(self.env.target).type(torch.FloatTensor))

        value, logit, (self.hx, self.cx), self.fcn = self.model(
            (Variable(self.state.unsqueeze(0)), (self.hx, self.cx), self.target))
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        self.entropy = -(log_prob * prob).sum(1)
        self.entropies.append(self.entropy)
        action = prob.multinomial().data
        log_prob = log_prob.gather(1, Variable(action))
        state, self.reward, self.done, self.info = self.env.step(
            int(action.cpu().numpy()))
        state = normalize_rgb_obs(state)
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1

        # print("self.fcn.shape: ", self.fcn.shape)
        # self.fcn.shape: torch.Size([1, 1, 90, 120])
        # depth.size(): torch.Size([1, 90, 120])

        # get_semantic.shape: (90, 120, 3)
        depth = self.env.get_depth()[:, :, 0].astype(float)
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.depth = Variable(torch.from_numpy(depth).type(torch.FloatTensor).cuda().unsqueeze(0))
        else:
            self.depth = Variable(torch.from_numpy(depth).type(torch.FloatTensor).unsqueeze(0))
        # print("depth.size(): ", self.depth.size()) # depth.size():  torch.Size([1, 90, 120])

        # print("self.fcn: ", self.fcn)
        # print("self.fcn.max(): ", self.fcn.max())
        # print("self.fcn.min(): ", self.fcn.min())

        # print("self.depth: ", self.depth)
        # print("self.depth.max(): ", self.depth.max())
        # print("self.depth.min(): ", self.depth.min())
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                depth_loss = F.l1_loss(self.fcn, self.depth, reduce=True, size_average=True)
        else:
            depth_loss = F.l1_loss(self.fcn, self.depth, reduce=True, size_average=True)
        # fcn_loss = cross_entropy2d(self.fcn, self.depth)
        # print("fcn_loss: ", fcn_loss)

        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        self.fcn_losses.append(depth_loss)

        return self

    def action_test(self):
        if self.done:
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    self.cx = Variable(torch.zeros(
                        1, 512).cuda(), volatile=True)
                    self.hx = Variable(torch.zeros(
                        1, 512).cuda(), volatile=True)
                    self.target = Variable(torch.from_numpy(self.env.target).type(torch.FloatTensor).cuda())
            else:
                self.cx = Variable(torch.zeros(1, 512), volatile=True)
                self.hx = Variable(torch.zeros(1, 512), volatile=True)
                self.target = Variable(torch.from_numpy(self.env.target).type(torch.FloatTensor))
        else:
            self.cx = Variable(self.cx.data, volatile=True)
            self.hx = Variable(self.hx.data, volatile=True)
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    self.target = Variable(torch.from_numpy(self.env.target).type(torch.FloatTensor).cuda())
            else:
                self.target = Variable(torch.from_numpy(self.env.target).type(torch.FloatTensor))

        value, logit, (self.hx, self.cx), fcn = self.model(
            (Variable(self.state.unsqueeze(0), volatile=True), (self.hx, self.cx), self.target))
        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()
        state, self.reward, self.done, self.info = self.env.step(action[0])
        state = normalize_rgb_obs(state)
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1

        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.fcn_losses = []
        return self


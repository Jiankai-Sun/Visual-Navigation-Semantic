from __future__ import division
import numpy as np
from cv2 import resize

from roomnav import RoomNavTask

from House3D import Environment, load_config, objrender
from constants import ACTION_SIZE, HOUSE_train, HOUSE_test, \
    OBSERVATION_SPACE_SHAPE, HOUSE


def atari_env(env_id, args, type):
    api = objrender.RenderAPI(w=OBSERVATION_SPACE_SHAPE[2], h=OBSERVATION_SPACE_SHAPE[1], device=args.gpu_ids[0])
    cfg = load_config('config_namazu.json')
    if type == 'train':
        pre_env = Environment(api, HOUSE_train[env_id], cfg, seed=args.seed)
        # TODO: temporarily use HOUSE
        # pre_env = Environment(api, HOUSE[env_id], cfg)
    elif type == 'test':
        pre_env = Environment(api, HOUSE_test[env_id], cfg, seed=args.seed)
        # TODO: temporarily use HOUSE
        # pre_env = Environment(api, HOUSE[env_id], cfg)
    env = RoomNavTask(pre_env, hardness=args.hardness, depth_signal=False, discrete_action=True,
                      max_steps=args.max_episode_length)

    return env

def _process_frame(frame, conf):
    frame = frame[conf["crop1"]:conf["crop2"] + 160, :160]
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = resize(frame, (80, conf["dimension2"]))
    frame = resize(frame, (80, 80))
    frame = np.reshape(frame, [1, 80, 80])
    return frame




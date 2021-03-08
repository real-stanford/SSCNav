from SCNav_agent import SCNavAgent
from utils.utils import d3_41_colors_rgb, ScalarMeanTracker

import torch

import random
import numpy as np
import argparse
import cv2
import shutil
import os
import copy
import json
from tensorboardX import SummaryWriter
from tqdm import tqdm

from train_agent import parser
import quaternion as nq

from habitat.tasks.nav.object_nav_task import (
            ObjectGoal,
                ObjectGoalNavEpisode,
                    ObjectViewLocation,
                    )

def main():
    args = parser.parse_args()

    #new_eval = args.new_eval
    new_eval = True
    fake_conf = args.fake_conf
    discrete = args.discrete
    att = args.att
    rc = args.rc
    unconf = args.unconf
    full_map = args.full_map
    args.max_step = 500
    
    
    args.config_paths = './configs/agent_test.yaml'

    args.save_dir = './result/exps/'


    save_dir = os.path.join(args.save_dir, args.title)
    if os.path.exists(save_dir):
        assert False, "Dir exists!"
    os.makedirs(save_dir)

    configs = None

    args.load_json = './data/val.json'

    if args.load_json != "":
        with open(args.load_json, 'r') as f:
            configs = json.load(f)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    agent = SCNavAgent(
            device = torch.device(args.device),
            config_paths = args.config_paths,
            flip = args.flip,
            save_dir = save_dir,
            #pano = bool(args.pano),
            pano = False,
            user_semantics = bool(args.user_semantics),
            seg_pretrained = args.seg_pretrained,
            cmplt = bool(args.cmplt),
            cmplt_pretrained = args.cmplt_pretrained,
            conf = bool(args.conf),
            conf_pretrained = args.conf_pretrained,
            targets = args.targets,
            aggregate = bool(args.aggregate),
            #aggregate = True,
            memory_size = args.memory_size,
            num_channel = args.num_channel,
            success_threshold = args.success_threshold,
            collision_threshold = args.collision_threshold,
            ignore = args.ignore,
            training = False,
            Q_pretrained = args.Q_pretrained,
            #Q_pretrained = '../result/1.pth',
            offset = args.offset,
            floor_threshold = args.floor_threshold,
            lr = args.lr,
            momentum = args.momentum,
            weight_decay = args.weight_decay,
            gamma = args.gamma,
            batch_size = args.batch_size,
            #batch_size = 1,
            buffer_size = args.buffer_size,
            height = args.height,
            area_x = args.area_x,
            area_z = args.area_z,
            h = args.h,
            w = args.w,
            h_new = args.h_new,
            w_new = args.w_new,
            max_step = args.max_step,
            #max_step = 5,
            navigable_base = args.navigable_base,
            success_reward = args.success_reward,
            step_penalty = args.step_penalty,
            approach_reward = args.approach_reward,
            collision_penalty = args.collision_penalty,
           # max_dist=args.max_dist,
            max_dist = float("inf"),
            scene_types = args.scene_types,
            double_dqn = bool(args.double_dqn),
            TAU = args.TAU,
            preconf=args.preconf,
            seg_threshold=args.seg_threshold,
            min_dist=0.,
            current_position = None,
            new_eval=new_eval,
            shortest=args.shortest,
            fake_conf=fake_conf,
            discrete=discrete,
            att=att,
            rc=rc,
            unconf=unconf,
            full_map = full_map)
    targets = copy.deepcopy(agent.targets)
    train_scalars = {}
    for target in targets + ['all']:
        train_scalars[target] = ScalarMeanTracker()



    max_test_epoch = len(list(configs.keys()))
    print("start to evaluate on %s episodes..." % max_test_epoch)
    pbar = tqdm(total=max_test_epoch)
    path_records = {}
    for ep_id in range(max_test_epoch):
        path_records[str(ep_id)] = {}
        if not new_eval:
            current_target = targets[ep_id % len(targets)]
            if new_eval and current_target in ['table', 
                'sofa', 'door']:
                continue
            if configs is None:
                agent.reset(current_target)
                while agent.best_path_length < 1.25 or agent.best_path_length > 25.:
                    agent.reset(current_target)
            else:
                agent.reset_config(configs[str(ep_id)])
        else:
            config = configs[str(ep_id)]
         #   config['start_position'][1]\
          #          = str(float(config['start_position'][1]) + 2.)
            current_target = config['target']
            agent.reset_config(config)

        
            

        path_records[str(ep_id)]['target'] = current_target
        path_records[str(ep_id)]['scene_id'] = agent.episode.scene_id
        path_records[str(ep_id)]['start_position']\
        = [str(x) for x in list(agent.env.sim.get_agent_state().position)]
        path_records[str(ep_id)]['start_rotation']\
        = [str(x) for x in
                list(nq.as_float_array(agent.env.sim.get_agent_state().rotation))]
        path_records[str(ep_id)]['best_path_length']\
                = str(agent.best_path_length)
        path_records[str(ep_id)]['actions'] = []

        step_id = 0
        innerbar = tqdm(total=args.max_step)
        while not agent.done:
            
            cv2.imwrite(os.path.join(save_dir, "%s_%s_obs_%s.png"
                    % (ep_id, step_id, agent.target)),
                    d3_41_colors_rgb[torch.argmax(agent.state[0,
                        :agent.num_channel,...] , dim=0)])

            if agent.user_semantics:
                cv2.imwrite(os.path.join(save_dir, "%s_%s_seg_%s.png"
                    % (ep_id, step_id, agent.target)),
                    d3_41_colors_rgb[agent.raw_semantics])
            if agent.conf:
                cv2.imwrite(os.path.join(save_dir, "%s_%s_conf_obs_%s.png"
                    % (ep_id, step_id, agent.target)),
                    agent.conf_obs[0,0].numpy() * 255.)
                    
            cv2.imwrite(os.path.join(save_dir, "%s_%s_rgb_%s.png"
                    % (ep_id, step_id, agent.target)),
                    agent.get_observations()['rgb'][..., [2, 1, 0]])

            
            
            tmp_depth = agent.get_observations()['depth'][..., 0]
            depth = tmp_depth * agent.d2x[..., 0]
            tmp_depth[depth == 0.] = 255.
            tmp_depth[(depth >0.) & (depth <= 1.)] = 122
            tmp_depth[depth > 1.] = 255.



            cv2.imwrite(os.path.join(save_dir, '%s_%s_dep_%s.png'
                % (ep_id, step_id, agent.target)),
                tmp_depth)
            if args.title not in ['random', 'randoms', 'random-seg',
                    'random_new']:
                dreward = agent.step(args.end_eps)
            else:
                dreward = agent.step(1.)

            if agent.user_semantics:
                cv2.imwrite(os.path.join(save_dir, "%s_%s_next_seg_%s.png"
                    % (ep_id, step_id, agent.target)),
                    d3_41_colors_rgb[agent.raw_semantics])
            tmp_depth = agent.get_observations()['depth'][..., 0]
            depth = tmp_depth * agent.d2x[..., 0]
            tmp_depth[depth == 0.] = 255.
            tmp_depth[(depth >0.) & (depth <= 1.)] = 122
            tmp_depth[depth > 1.] = 255.



            cv2.imwrite(os.path.join(save_dir, '%s_%s_next_dep_%s.png'
                % (ep_id, step_id, agent.target)),
                tmp_depth)
            if not discrete:
                path_records[str(ep_id)]['actions'].append((int(agent.action[0]),
                int(agent.action[1])))

                cmap = agent.q_map.numpy()
                if np.max(cmap) == np.min(cmap):
                    cmap = np.zeros(cmap.shape).astype(np.uint8)
                else:
                    cmap = (cmap - np.min(cmap)) / (np.max(cmap)
                                - np.min(cmap)) * 255.
                    cmap = cmap.astype(np.uint8)
                cmap = cv2.cvtColor(cmap, cv2.COLOR_GRAY2BGR)
                cmap = cv2.applyColorMap(cmap, cv2.COLORMAP_JET)
                cv2.circle(cmap, (agent.action[1], agent.action[0]), 3, (20, 20, 20), -1)
                cv2.imwrite(os.path.join(save_dir, "%s_%s_q_map_%s.png"
                    % (ep_id, step_id, agent.target)),
                    cmap)
                
            while step_id < agent.eps_len:
                step_id += 1
                innerbar.update(1)

        
        path_records[str(ep_id)]['path_length'] = str(agent.path_length)
        path_records[str(ep_id)]['success'] = str(agent.success)
        path_records[str(ep_id)]['reward'] = str(agent.reward)
        path_records[str(ep_id)]['eps_len'] = str(agent.eps_len)
        path_records[str(ep_id)]['action_list'] = [str(t) for t in
                agent.action_list]
        innerbar.close()
        pbar.update(1)


        torch.cuda.empty_cache() 
        results = {
               "path_length": agent.path_length,
               "reward": agent.reward,
               "success": int(agent.success),
               "eps_len": agent.eps_len,
               "SPL": int(agent.success) * agent.best_path_length\
                       / max(agent.path_length, agent.best_path_length)}

        train_scalars['all'].add_scalars(results)
        train_scalars[current_target].add_scalars(results)
      



    data = {}
    for cat in (targets + ['all']):
        data[cat] = train_scalars[cat].pop_and_reset()
    with open(os.path.join(save_dir, 'data.json'), 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)
    with open(os.path.join(save_dir, "vis.json"), 'w') as fp:
        print(path_records)
        json.dump(path_records, fp)
    pbar.close()


if __name__ == "__main__":

    main()

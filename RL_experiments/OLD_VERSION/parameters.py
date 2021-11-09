import numpy as np
from math import pi

SETUP = {

    'K'                         : 2,
    'B'                         : 4,
    'M'                         : 2,
    'N'                         : 64,
    'group_size'                : 16,
    # controllable elements: (48/16) = 3 => *2 = 6
    'kappa_H'                   : 13,
    'kappa_G'                   : 10,#13.2, # 13
    'frequency'                 : 32000000000.,
    'RIS_resolution'            : 2,
    'RIS_phases'                : np.array([0, pi]),
    'codebook_rays_per_RX'      : 4,
    'noise_variance'            : 80,
    'transmit_power'            : 1.,
    'BS_position'               : np.array([10., 0, 2]),
    'RIS_positions'             : np.array([[5, 25, 2.], [15, 25, 2.]]),
    'RX_box'                    : np.array([[5, 20, 1.0], [15, 30, 2.0]]),
    'direct_link_attenuation'   : 100,
    'observation_noise_variance': 0,
    'ignore_direct_link'        : True,

}




DQN_PARAMS = {
    'fc_layer_params'            : [100, 100],
    'num_iterations'             : 100,         # multiplied by number of actions
    'initial_collect_steps'      : 1000,
    'collect_steps_per_iteration': 1,
    'replay_buffer_max_length'   : 10000,
    'batch_size'                 : 32,
    'learning_rate'              : 10e-3,
    'log_interval'               : 5,
    'eval_interval'              : 10,
    'epsilon_greedy'             : 0.2,
    'gradient_clipping'          : 100.,
    'n_step_update'              : 1,
    'target_update_tau'          : 1.,
    'target_update_period'       : 5,
    'gamma'                      : 0.9999,
    'num_eval_episodes'          : 5,
    'num_actions'                : None, # to be inserted by code
}


# NEURAL_LIN_UCB_PARAMS = {
#     'fc_layer_params'                  : (100, 100),
#     'encoding_dim'                     : 32,
#     'num_iterations'                   : 100, # multiplied by number of actions
#     'steps_per_loop'                   : 10,
#     'batch_size'                       : 1,
#     'log_interval'                     : 5,
#     'eval_interval'                    : 5,
#     'learning_rate'                    : 1e-3,
#     'encoding_network_num_train_steps' : 0.1, # fraction of the final value of num_iterations
#     'epsilon_greedy'                   : 0.2,
#     'alpha'                            : 5.,
#     'gamma'                            : 0.5,
#     'num_eval_episodes'                : 1000,
#     'gradient_clipping'                : 1.,
#     'num_actions'                      : None, # to be inserted by code
# }


NEURAL_EPSILON_GREEDY_PARAMS ={
    'fc_layer_params'                  : (32, 32,),
    'dropout_p'                        : 0.05,
    'kernel_l2_reg'                    : 1e-07,
    'initialization_variance_scale'    : 10,
    'num_iterations'                   : 150, # multiplied by number of actions
    'steps_per_loop'                   : 1,
    'batch_size'                       : 1,
    'log_interval'                     : 5,
    'eval_interval'                    : 1000,
    'learning_rate'                    : 2e-3,
    'epsilon_greedy'                   : 0.3,
    'num_eval_episodes'                : 500,
    'gradient_clipping'                : 5,
    'num_actions'                      : None, # to be inserted by code
}

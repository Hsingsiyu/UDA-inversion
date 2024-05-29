"""
This file runs the main training/val loop
"""
import json
import os
import pprint
import sys

import torch

sys.path.append(".")
sys.path.append("..")

from datetime import datetime

from options.train_options import TrainOptions
# from training.coach import Coach
from training.coach_fdal import Coach


def main():
	opts = TrainOptions().parse()
	create_initial_experiment_dir(opts)
	coach = Coach(opts)
	coach.train()
 

def create_initial_experiment_dir(opts):
	current_time = datetime.now().strftime('%b%d_%H-%M')
	real_exp_dir=os.path.join(opts.exp_dir,current_time+opts.dataset_type+str(opts.batch_size))
	os.makedirs(real_exp_dir)
	opts.exp_dir=real_exp_dir
	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)
 

if __name__ == '__main__':
	main()

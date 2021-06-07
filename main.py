import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import StaticSparseTD3
import DS_TD3
import datetime

printComments=True

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	if (printComments):
		print("---------------------------------------")
		print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
		print("---------------------------------------")

	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="DS-TD3")      # Policy name (TD3, DS-TD3 or StaticSparseTD3)
	parser.add_argument("--env", default="HalfCheetah-v3")          # OpenAI gym environment name
	parser.add_argument("--seed", default=2, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=100, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", default=True)        		# Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

	parser.add_argument("--save_model_period", default=1e5, type=int)  # Save model and optimizer parameters after the set number of iterations
	parser.add_argument("--ann_noHidNeurons", default=256, type=int)  #
	parser.add_argument("--ann_epsilonHid1", default=7, type=int)  #   lambda 1
	parser.add_argument("--ann_epsilonHid2", default=64, type=int)  #  lambda 2
	parser.add_argument("--ann_setZeta", default=0.05)  #
	parser.add_argument("--ann_ascTopologyChangePeriod", default=1e3, type=int)  #
	parser.add_argument("--ann_earlyStopTopologyChange", default=5e4, type=int)  #

	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}_{args.ann_epsilonHid1}_{args.ann_epsilonHid2}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		"noHidNeurons":args.ann_noHidNeurons,
	}

	# Target policy smoothing is scaled wrt the action scale
	kwargs["policy_noise"] = args.policy_noise * max_action
	kwargs["noise_clip"] = args.noise_clip * max_action
	kwargs["policy_freq"] = args.policy_freq

	# Initialize policy
	if args.policy == "TD3":
		policy = TD3.TD3(**kwargs)
	elif args.policy == "StaticSparseTD3":
		kwargs["epsilonHid1"]=args.ann_epsilonHid1
		kwargs["epsilonHid2"]=args.ann_epsilonHid2
		policy = StaticSparseTD3.StaticSparseTD3(**kwargs)
	elif args.policy == "DS-TD3":
		kwargs["epsilonHid1"]=args.ann_epsilonHid1
		kwargs["epsilonHid2"]=args.ann_epsilonHid2
		kwargs["setZeta"]=args.ann_setZeta
		kwargs["ascTopologyChangePeriod"]=args.ann_ascTopologyChangePeriod
		kwargs["earlyStopTopologyChangeIteration"] = args.max_timesteps-args.start_timesteps-args.ann_earlyStopTopologyChange
		policy = DS_TD3.SETSparseTD3(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	if args.save_model:
		policy.save(f"./models/{file_name}_iter_0")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	t1 = datetime.datetime.now()
	tin= datetime.datetime.now()
	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			if (printComments):
				print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Time: {datetime.datetime.now()-t1}")
			t1 = datetime.datetime.now()
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/{file_name}", evaluations)

		if args.save_model:
			if (t+1) % args.save_model_period == 0:
				policy.save(f"./models/{file_name}_iter_{t+1}")

	print("Total running time",datetime.datetime.now()-tin)

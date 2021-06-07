import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sparse_utils as sp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action,noHidNeurons,epsilonHid1,epsilonHid2):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, noHidNeurons)
		[self.noPar1, self.mask1] = sp.initializeEpsilonWeightsMask("actor first layer", epsilonHid1, state_dim, noHidNeurons)
		self.torchMask1=torch.from_numpy(self.mask1).float().to(device)
		self.l1.weight.data.mul_(torch.from_numpy(self.mask1).float())

		self.l2 = nn.Linear(noHidNeurons, noHidNeurons)
		[self.noPar2, self.mask2] = sp.initializeEpsilonWeightsMask("actor second layer", epsilonHid2, noHidNeurons, noHidNeurons)
		self.torchMask2 = torch.from_numpy(self.mask2).float().to(device)
		self.l2.weight.data.mul_(torch.from_numpy(self.mask2).float())

		self.l3 = nn.Linear(noHidNeurons, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim,noHidNeurons,epsilonHid1,epsilonHid2):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, noHidNeurons)
		[self.noPar1, self.mask1] = sp.initializeEpsilonWeightsMask("critic Q1 first layer", epsilonHid1, state_dim + action_dim, noHidNeurons)
		self.torchMask1=torch.from_numpy(self.mask1).float().to(device)
		self.l1.weight.data.mul_(torch.from_numpy(self.mask1).float())

		self.l2 = nn.Linear(noHidNeurons, noHidNeurons)
		[self.noPar2, self.mask2] = sp.initializeEpsilonWeightsMask("critic Q1 second layer", epsilonHid2, noHidNeurons, noHidNeurons)
		self.torchMask2 = torch.from_numpy(self.mask2).float().to(device)
		self.l2.weight.data.mul_(torch.from_numpy(self.mask2).float())

		self.l3 = nn.Linear(noHidNeurons, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, noHidNeurons)
		[self.noPar4, self.mask4] = sp.initializeEpsilonWeightsMask("critic Q2 first layer", epsilonHid1, state_dim + action_dim, noHidNeurons)
		self.torchMask4 = torch.from_numpy(self.mask4).float().to(device)
		self.l4.weight.data.mul_(torch.from_numpy(self.mask4).float())

		self.l5 = nn.Linear(noHidNeurons, noHidNeurons)
		[self.noPar5, self.mask5] = sp.initializeEpsilonWeightsMask("critic Q2 second layer", epsilonHid2, noHidNeurons, noHidNeurons)
		self.torchMask5 = torch.from_numpy(self.mask5).float().to(device)
		self.l5.weight.data.mul_(torch.from_numpy(self.mask5).float())

		self.l6 = nn.Linear(noHidNeurons, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class StaticSparseTD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		noHidNeurons=256,
		epsilonHid1=20,
		epsilonHid2=20,
	):

		self.actor = Actor(state_dim, action_dim, max_action,noHidNeurons,epsilonHid1,epsilonHid2).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3, weight_decay=0.0002)

		self.critic = Critic(state_dim, action_dim,noHidNeurons,epsilonHid1,epsilonHid2).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=0.0002)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=100):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Maintain the same sparse connectivity for critic
		self.critic.l1.weight.data.mul_(self.critic.torchMask1)
		self.critic.l2.weight.data.mul_(self.critic.torchMask2)
		self.critic.l4.weight.data.mul_(self.critic.torchMask4)
		self.critic.l5.weight.data.mul_(self.critic.torchMask5)

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Maintain the same sparse connectivity for actor
			self.actor.l1.weight.data.mul_(self.actor.torchMask1)
			self.actor.l2.weight.data.mul_(self.actor.torchMask2)

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def print_sparsity(self):
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			if(len(target_param.shape)>1):
				critic_current_sparsity =  ((target_param==0).sum().cpu().data.numpy()*1.0/(target_param.shape[0]*target_param.shape[1]))
				print("target critic sparsity", critic_current_sparsity)
				
				critic_current_sparsity =  ((param==0).sum().cpu().data.numpy()*1.0/(param.shape[0]*param.shape[1]))
				print("critic sparsity", critic_current_sparsity)
		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			if(len(target_param.shape)>1):
				critic_current_sparsity =  ((target_param==0).sum().cpu().data.numpy()*1.0/(target_param.shape[0]*target_param.shape[1]))
				print("target actor sparsity", critic_current_sparsity)

				critic_current_sparsity =  ((param==0).sum().cpu().data.numpy()*1.0/(param.shape[0]*param.shape[1]))
				print("actor sparsity", critic_current_sparsity)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		
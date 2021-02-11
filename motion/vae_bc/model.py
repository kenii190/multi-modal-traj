import numpy as np
import torch
import torch.nn as nn
from distrib import Categorical, DiagGaussian
from utils import init


#Policy network
class PolicyNet(nn.Module):
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(self, s_dim, a_dim, c_dim, conti=False):
		super(PolicyNet, self).__init__()
		self.conti = conti

		init_ = lambda m: init(
			m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0), 
			nn.init.calculate_gain('relu')
		)
		self.main = nn.Sequential(
			init_(nn.Linear(s_dim+c_dim, 128)),
			nn.ReLU(),
            init_(nn.Linear(128, 128)),
            nn.ReLU()
		)

		if conti:
			self.dist = DiagGaussian(128, a_dim)
		else:
			self.dist = Categorical(128, a_dim)

	#-----------------------
	# Forward
	#-----------------------
	def forward(self, ob, c, deterministic=False):
		feature = self.main(torch.cat((ob, c), 1))
		dist    = self.dist(feature)

		if deterministic:
			action = dist.mode()
		else:
			action = dist.sample()

		if self.conti:
			return action, dist.log_probs(action)
		
		return action[:, 0], dist.log_probs(action)

	#-----------------------
	# Output action
	#-----------------------
	def action_step(self, ob, c, deterministic=True):
		feature = self.main(torch.cat((ob, c), 1))
		dist    = self.dist(feature)

		if deterministic:
			action = dist.mode()
		else:
			action = dist.sample()

		if self.conti:
			return action

		return action[:, 0]

	#-----------------------
	# Evaluate log-probs & entropy
	#-----------------------
	def evaluate(self, ob, action, c):
		feature = self.main(torch.cat((ob, c), 1))
		dist    = self.dist(feature)

		return dist.log_probs(action), dist.entropy()


#RNN Encoder
class RNNEncoder(nn.Module):
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(self, sa_dim, z_dim, h_dim=128, n_layer=2, bidirectional=True):
		super(RNNEncoder, self).__init__()
		self.sa_dim  = sa_dim
		self.h_dim   = h_dim
		self.z_dim   = z_dim
		self.n_layer = n_layer
		self.n_dir   = 2 if bidirectional else 1

		self.gru  = nn.GRU(sa_dim, h_dim, num_layers=n_layer, bidirectional=bidirectional)
		self.fc_z = nn.Linear(h_dim*self.n_layer*self.n_dir, z_dim*2)

	#-----------------------
	# Forward
	#-----------------------
	def forward(self, traj, seq_lens):
		#traj: (seq_len, mb_size, sa_dim)
		#h   : (n_layer*n_dir, mb_size, h_dim)
		pack   = torch.nn.utils.rnn.pack_padded_sequence(traj, seq_lens, enforce_sorted=False)
		out, h = self.gru(traj, None)

		#h: (mb_size, z_dim*2)
		h = self.fc_z(h.view(-1, self.h_dim*self.n_layer*self.n_dir))
		z_mean   = h[:, :self.z_dim]
		z_logstd = h[:, self.z_dim:]

		#Reparameterization
		eps = torch.randn_like(z_mean)
		z_sample = eps.mul(torch.exp(z_logstd)) + z_mean

		#z: (mb_size, z_dim)
		return z_mean, z_logstd, z_sample


	#-----------------------
	# Get latent code
	#-----------------------
	def get_code(self, traj, deterministic=True):
		#traj: (seq_len, mb_size, sa_dim)
		#h   : (n_layer*n_dir, mb_size, h_dim)
		out, h = self.gru(traj, None)

		#h: (mb_size, z_dim*2)
		h = self.fc_z(h.view(-1, self.h_dim*self.n_layer*self.n_dir))
		z_mean   = h[:, :self.z_dim]
		z_logstd = h[:, self.z_dim:]

		#z: (mb_size, z_dim)
		if deterministic:
			return z_mean

		#Reparameterization
		eps = torch.randn_like(z_mean)
		z_sample = eps.mul(torch.exp(z_logstd)) + z_mean

		return z_sample
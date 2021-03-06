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


#Value network
class ValueNet(nn.Module):
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(self, s_dim, c_dim):
		super(ValueNet, self).__init__()

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
		self.fc_v = init_(nn.Linear(128, 1))

	#-----------------------
	# Forward
	#-----------------------
	def forward(self, ob, c):
		feature = self.main(torch.cat((ob, c), 1))
		value   = self.fc_v(feature)

		return value[:, 0]


#Encoder Network
class EncoderNet(nn.Module):
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(self, sa_dim, c_dim):
		super(EncoderNet, self).__init__()
		self.c_dim = c_dim

		init_ = lambda m: init(
			m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0), 
			nn.init.calculate_gain('relu')
		)
		self.fc1 = nn.Sequential(
			init_(nn.Linear(sa_dim, 128)),
			nn.ReLU(),
            init_(nn.Linear(128, 128)),
            nn.ReLU(),
            init_(nn.Linear(128, c_dim*2))
		)

	#-----------------------
	# Forward
	#-----------------------
	def forward(self, sa, deterministic=False):
		h = self.fc1(sa)
		c_mean   = h[:, :self.c_dim]
		c_logstd = h[:, self.c_dim:]

		#Reparameterization
		eps = torch.randn_like(c_mean)
		c_sample = eps.mul(torch.exp(c_logstd)) + c_mean
		
		return c_mean, c_logstd, c_sample

	#-----------------------
	# Get latent code
	#-----------------------
	def get_code(self, sa, deterministic=False):
		h = self.fc1(sa)
		c_mean   = h[:, :self.c_dim]
		c_logstd = h[:, self.c_dim:]

		if deterministic:
			return c_mean

		#Reparameterization
		eps = torch.randn_like(c_mean)
		c_sample = eps.mul(torch.exp(c_logstd)) + c_mean
		
		return c_sample


#Discriminator Network
class DiscriminatorNet(nn.Module):
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(self, sa_dim, c_dim):
		super(DiscriminatorNet, self).__init__()

		init_ = lambda m: init(
			m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0), 
			nn.init.calculate_gain('relu')
		)
		self.fc1 = nn.Sequential(
			init_(nn.Linear(sa_dim+c_dim, 128)),
			nn.ReLU(),
            init_(nn.Linear(128, 128)),
            nn.ReLU(),
            init_(nn.Linear(128, 1))
		)
	
	#-----------------------
	# Forward
	#-----------------------
	def forward(self, sa, c):
		logits = self.fc1(torch.cat((sa, c), 1))
		return torch.sigmoid(logits)

	#-----------------------
	# Get prob. that (s, a) is real
	#-----------------------
	def get_prob(self, sa, c):
		logits = self.fc1(torch.cat((sa, c), 1))
		return torch.sigmoid(logits)
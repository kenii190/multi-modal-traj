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
	def __init__(self, s_dim):
		super(ValueNet, self).__init__()

		init_ = lambda m: init(
			m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0), 
			nn.init.calculate_gain('relu')
		)
		self.main = nn.Sequential(
			init_(nn.Linear(s_dim, 128)),
			nn.ReLU(),
            init_(nn.Linear(128, 128)),
            nn.ReLU()
		)
		self.fc_v = init_(nn.Linear(128, 1))

	#-----------------------
	# Forward
	#-----------------------
	def forward(self, ob):
		feature = self.main(ob)
		value   = self.fc_v(feature)

		return value[:, 0]


#Discriminator Network
class DiscriminatorNet(nn.Module):
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(self, sa_dim, z_dim=8):
		super(DiscriminatorNet, self).__init__()
		self.z_dim = z_dim

		init_ = lambda m: init(
			m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0), 
			nn.init.calculate_gain('relu')
		)
		self.fc_z = nn.Sequential(
			init_(nn.Linear(sa_dim, 128)),
			nn.ReLU(),
            init_(nn.Linear(128, 128)),
            nn.ReLU(),
            init_(nn.Linear(128, z_dim*2))
		)
		self.fc_d = init_(nn.Linear(z_dim, 1))
	
	#-----------------------
	# Forward
	#-----------------------
	def forward(self, sa):
		h = self.fc_z(sa)
		z_mean   = h[:, :self.z_dim]
		z_logstd = h[:, self.z_dim:]

		#Reparameterization
		z_std    = torch.exp(z_logstd)
		eps      = torch.randn_like(z_std)
		z_sample = eps.mul(z_std) + z_mean

		logits  = self.fc_d(z_sample)

		return torch.sigmoid(logits), z_mean, z_logstd

	#-----------------------
	# Get prob. that (s, a) is real
	#-----------------------
	def get_prob(self, sa):
		h = self.fc_z(sa)
		z_mean   = h[:, :self.z_dim]
		z_logstd = h[:, self.z_dim:]

		#Reparameterization
		z_std    = torch.exp(z_logstd)
		eps      = torch.randn_like(z_std)
		z_sample = eps.mul(z_std) + z_mean

		logits  = self.fc_d(z_sample)

		return torch.sigmoid(logits)


#Encoder Network
class EncoderNet(nn.Module):
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(self, sa_dim, c_dim, conti=False):
		super(EncoderNet, self).__init__()
		self.conti = conti

		init_ = lambda m: init(
			m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0), 
			nn.init.calculate_gain('relu')
		)
		self.main = nn.Sequential(
			init_(nn.Linear(sa_dim, 128)),
			nn.ReLU(),
            init_(nn.Linear(128, 128)),
            nn.ReLU()
		)

		if conti:
			self.dist = DiagGaussian(128, c_dim)
		else:
			self.dist = Categorical(128, c_dim)

	#-----------------------
	# Forward
	#-----------------------
	def forward(self, sa, deterministic=False):
		feature = self.main(sa)
		dist    = self.dist(feature)

		if deterministic:
			c = dist.mode()
		else:
			c = dist.sample()

		if self.conti:
			return c, dist.log_probs(c)
		
		return c[:, 0], dist.log_probs(c)

	#-----------------------
	# Get latent code
	#-----------------------
	def get_code(self, sa, deterministic=True):
		feature = self.main(sa)
		dist    = self.dist(feature)

		if deterministic:
			c = dist.mode()
		else:
			c = dist.sample()

		if self.conti:
			return c
		
		return c[:, 0]

	#-----------------------
	# Get log-prob
	#-----------------------
	def get_logp(self, sa, c):
		feature = self.main(sa)
		dist    = self.dist(feature)

		return dist.log_probs(c)
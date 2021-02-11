import torch
import torch.nn as nn
import numpy as np


def reparameterize(mean, logstd):
	std = logstd.exp()
	eps = torch.autograd.Variable(std.data.new(std.size()).normal_())

	if mean.is_cuda:
		eps = eps.cuda()

	return eps.mul(std).add(mean)


def init(module, weight_init, bias_init, gain=1):
	weight_init(module.weight.data, gain=gain)
	bias_init(module.bias.data)

	return module


#Encoder module
class Encoder(nn.Module):
	def __init__(self, inp_dim, z_dim, h_dim=256):
		super(Encoder, self).__init__()
		self.z_dim = z_dim

		init_ = lambda m: init(
			m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0),
			np.sqrt(2)
		)
		self.fc1 = nn.Sequential(
			init_(nn.Linear(inp_dim, h_dim)),
			nn.ReLU()
		)
		self.res1 = nn.Sequential(
			init_(nn.Linear(h_dim, h_dim)),
			nn.ReLU(),
			init_(nn.Linear(h_dim, h_dim)),
			nn.ReLU(),
		)
		self.res2 = nn.Sequential(
			init_(nn.Linear(h_dim, h_dim)),
			nn.ReLU(),
			init_(nn.Linear(h_dim, h_dim)),
			nn.ReLU(),
		)
		self.fc_out = init_(nn.Linear(h_dim, z_dim*2))


	def forward(self, inp):
		h = self.fc1(inp)
		h = self.res1(h) + h
		h = self.res2(h) + h
		h = self.fc_out(h)

		return h[:, :self.z_dim], h[:, self.z_dim:]


	def mean(self, inp):
		h = self.fc1(inp)
		h = self.res1(h) + h
		h = self.res2(h) + h
		h = self.fc_out(h)

		return h[:, :self.z_dim]


	def sample(self, inp):
		h = self.fc1(inp)
		h = self.res1(h) + h
		h = self.res2(h) + h
		h = self.fc_out(h)

		z_mean   = h[:, :self.z_dim]
		z_logstd = h[:, self.z_dim:]

		return reparameterize(z_mean, z_logstd)


#Decoder module
class Decoder(nn.Module):
	def __init__(self, out_dim, z_dim, h_dim=256):
		super(Decoder, self).__init__()

		init_ = lambda m: init(
			m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0),
			np.sqrt(2)
		)

		self.fc1 = nn.Sequential(
			init_(nn.Linear(z_dim, h_dim)),
			nn.ReLU()
		)
		self.res1 = nn.Sequential(
			init_(nn.Linear(h_dim, h_dim)),
			nn.ReLU(),
			init_(nn.Linear(h_dim, h_dim)),
			nn.ReLU(),
		)
		self.res2 = nn.Sequential(
			init_(nn.Linear(h_dim, h_dim)),
			nn.ReLU(),
			init_(nn.Linear(h_dim, h_dim)),
			nn.ReLU(),
		)
		self.fc_out = nn.Sequential(
			init_(nn.Linear(h_dim, out_dim)),
			nn.Sigmoid()
		)


	def forward(self, z):
		h = self.fc1(z)
		h = self.res1(h) + h
		h = self.res2(h) + h
		h = self.fc_out(h)

		return h
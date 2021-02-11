import torch
import torch.nn as nn
from utils import init


#Adding bias module
class AddBias(nn.Module):
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(self, bias):
		super(AddBias, self).__init__()
		self._bais = nn.Parameter(bias.unsqueeze(1))

	#-----------------------
	# Forward
	#-----------------------
	def forward(self, x):
		if x.dim() == 2:
			bias = self._bais.t().view(1, -1)
		else:
			bias = self._bais.t().view(1, -1, 1, 1)

		return x + bias


#Categorical distribution module
class FixedCategorical(torch.distributions.Categorical):
	#-----------------------
	# Sample
	#-----------------------
	def sample(self):
		return super().sample().unsqueeze(-1)

	#-----------------------
	# Log-probability
	#-----------------------
	def log_probs(self, actions):
		return super() \
			.log_prob(actions.squeeze(-1)) \
			.view(actions.size(0), -1) \
			.sum(-1)

	#-----------------------
	# Mode
	#-----------------------
	def mode(self):
		return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(self, n_inp, n_out):
		super(Categorical, self).__init__()

		init_ = lambda m: init(
			m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0),
			gain=0.01
		)
		self.linear = nn.Sequential(
			init_(nn.Linear(n_inp, n_out)),
			nn.Softmax(dim=1)
		)

	#-----------------------
	# Forward
	#-----------------------
	def forward(self, x):
		return FixedCategorical(logits=self.linear(x))


#Diagonal Gaussian distribution module
class FixedNormal(torch.distributions.Normal):
	#-----------------------
	# Log-probability
	#-----------------------
	def log_probs(self, actions):
		return super().log_prob(actions).sum(-1)

	#-----------------------
	# Entropy
	#-----------------------
	def entropy(self):
		return super().entropy().sum(-1)

	#-----------------------
	# Mode
	#-----------------------
	def mode(self):
		return self.mean


class DiagGaussian(nn.Module):
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(self, n_inp, n_out, use_logstd=False):
		super(DiagGaussian, self).__init__()
		self.use_logstd = use_logstd
		self.n_out = n_out

		init_ = lambda m: init(
			m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0)
		)

		if self.use_logstd:
			self.fc_main = init_(nn.Linear(n_inp, n_out*2))
		else:
			self.fc_main  = init_(nn.Linear(n_inp, n_out))
			self.b_logstd = AddBias(torch.zeros(n_out))

	#-----------------------
	# Forward
	#-----------------------
	def forward(self, x):
		if self.use_logstd:
			h = self.fc_main(x)
			mean   = h[:, :self.n_out]
			logstd = h[:, self.n_out:]
		else:
			mean = self.fc_main(x)
			logstd = self.b_logstd(torch.zeros_like(mean))

		return FixedNormal(mean, logstd.exp())
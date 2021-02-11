import numpy as np
import pickle as pkl


class MotionEnv():
	#-------------------------
	# Constructor
	#-------------------------
	def __init__(self, data_path, z_dim=8, max_step=512):
		self.s_traj, _ = pkl.load(open(data_path, "rb"))
		self.s_dim     = z_dim * 4
		self.a_dim     = z_dim
		self.z_dim     = z_dim
		self.state     = np.zeros([4, self.z_dim], dtype=np.float32)
		self.max_step  = max_step
		self.n_step    = 1


	#-------------------------
	# Step
	#-------------------------
	def step(self, action):
		pos_prev = self.state[3]
		pos_cur  = pos_prev + action

		for i in range(3):
			self.state[i, :] = self.state[i+1, :]
			
		self.state[3, :] = pos_cur
		self.n_step += 1

		done   = False
		reward = 0.0

		if np.linalg.norm(pos_cur) > 6.0:
			reward -= 30.0
			done    = True

		if self.n_step >= self.max_step:
			done = True

		return np.copy(self.state.flatten()), reward, done, 0


	#-------------------------
	# Reset
	#-------------------------
	def reset(self):
		self.state[:] = self.s_traj[np.random.randint(0, len(self.s_traj))][np.random.randint(0, 8)].reshape(4, self.z_dim)
		self.n_step   = 1

		return np.copy(self.state.flatten())


	#-------------------------
	# Close
	#-------------------------
	def close(self):
		del self.s_traj


#-------------------------
# Make an environment
#-------------------------
def make(data_path, max_step=512):
	return MotionEnv(data_path, max_step=max_step)
import math
import numpy as np
import pickle as pkl
from traffic_env import TrafficEnv
from tqdm import tqdm


#-----------------------
# Circle equation
#-----------------------
def circle(theta, r):
	return r*math.cos(theta), r*math.sin(theta)


#-----------------------
# Go straight
#-----------------------
def policy_straight(p, p_tar):
	dx   = p_tar[0] - p[0]
	dy   = p_tar[1] - p[1]
	norm = math.sqrt(dx*dx + dy*dy)

	return [dx / norm, dy / norm] if norm > 1e-8 else [0, 0]


#-----------------------
# Turn (counter)clockwise
#-----------------------
def policy_turn(p, r, center, clockwise=True):
	if clockwise:
		theta = math.atan2(p[1]-center[1], p[0]-center[0]) - 2 * math.asin(0.005 / r)
	else:
		theta = math.atan2(p[1]-center[1], p[0]-center[0]) + 2 * math.asin(0.005 / r)

	p_tar = circle(theta, r)
	dx    = p_tar[0] + center[0] - p[0]
	dy    = p_tar[1] + center[1] - p[1]

	norm  = math.sqrt(dx*dx + dy*dy)

	return [dx / norm, dy / norm] if norm > 1e-8 else [0, 0]


#-----------------------
# Expert policy 1, 2, 3
#-----------------------
def expert_policy(state, n=0):
	#Expert 0
	if n == 0:
		if state[-2] > 0.16:
			if state[-1] < 0.3:
				return policy_turn(state[-2:], 0.535, [0.0, 0.5], clockwise=False)
			else:
				return policy_straight(state[-2:], [0.6, 0.6])
		else:
			return policy_straight(state[-2:], [1.1, 0.0])

	#Expert 1
	elif n == 1:
		if state[-2] > 0.16:
			if state[-1] > -0.3:
				return policy_turn(state[-2:], 0.535, [0.0, -0.5], clockwise=True)
			else:
				return policy_straight(state[-2:], [0.6, -0.6])
		else:
			return policy_straight(state[-2:], [1.1, 0.0])

	#Expert 2
	return policy_straight(state[-2:], [1.1, 0.0])


#-----------------------
# Main
#-----------------------
def main():
	env       = TrafficEnv()
	n_episode = 1024
	s_traj    = []
	a_traj    = []
	idx       = 0

	for i in range(3):
		for j in tqdm(range(n_episode)):
			state = env.reset()
			s_traj.append([])
			a_traj.append([])

			while True:
				#env.render()
				action = expert_policy(state, i)
				
				s_traj[idx].append(state)
				a_traj[idx].append(action)

				state, reward, done, info = env.step(action)

				if done:
					s_traj[idx] = np.array(s_traj[idx], dtype=np.float32)
					a_traj[idx] = np.array(a_traj[idx], dtype=np.float32)
					idx += 1
					break

	pkl.dump((s_traj, a_traj), open("./expert_traj.pkl", "wb"))


if __name__ == '__main__':
	main()
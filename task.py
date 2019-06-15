import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 1
        self.action_high = 1000
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
    
    def dim_chan(self, pos, tar_pos, index):
        return np.abs(pos[index] - tar_pos[index])
    
    def check_reward(self, dist, ty='x'):
        reward = 0
        if ty != 'z':
            if dist > 0 and dist <= 5:
                reward += dist*400
            else:
                reward -= dist*200
        elif ty == 'z':
            if dist > 0 and dist <= 15:
                reward += dist*400
            elif dist > 15:
                reward -= dist*200
            elif dist <= 0:
                reward -= dist*500
        reward = np.tanh(reward)
        return reward
        
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        position = self.sim.pose[:3]
        reward = 0
        #reward = 1.-.3*(abs(position - self.target_pos)).sum()
        #distance = np.abs(np.sqrt(sum(list(map(lambda a: a**2, list(map(lambda a, b: a - b, position, self.target_pos)))))))
        x = self.dim_chan(position, self.target_pos, 0)
        y = self.dim_chan(position, self.target_pos, 1)
        z = self.dim_chan(position, self.target_pos, 2)
        reward += self.check_reward(x)
        reward += self.check_reward(y)
        reward += self.check_reward(z, ty='z')
        if self.sim.time > self.sim.runtime:           
            reward -= 100
            self.sim.done = True
        return reward 

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
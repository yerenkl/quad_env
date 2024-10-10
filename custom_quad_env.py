import pybullet as p
import numpy as np
import time
import pybullet_utils.bullet_client as bc
import random
from scipy.spatial.transform import Rotation as R
from gymnasium import Env, spaces, make
import pandas as pd

import time

SIM_TIME = 1/240
default_pos = [0.037199, 0.660252, -1.200187, -0.028954, 0.618814, -1.183148, 0.048225,0.690008,-1.254787,-0.050525,0.661355,-1.243304]
grav_vec = np.array([0,0,-1])
sit_pos = [0.0, 1.10, -2.697, 0.0, 1.10, -2.697, 0.0, 1.10, -2.697, 0.0, 1.10, -2.697]

def gaussian_rbf(x, y, c):
    distance_squared = np.sum((x - y)**2)
    return np.exp(distance_squared*c)

def normalizer(x):
    var = np.exp(x)/(1+np.exp(x))
    return np.interp(var, [0, 1], [-1, 1])

def quaternion_randomizer():
    x0 = random.uniform(0,1)
    x1 = random.uniform(0,1)
    x2 = random.uniform(0,1)
    theta1 = 2*np.pi*x1
    theta2 = 2*np.pi*x2
    s1 = np.sin(theta1)
    s2 = np.sin(theta2)
    c1 = np.cos(theta1)
    c2 = np.cos(theta2)
    r1 = np.sqrt(1-x0)
    r2 = np.sqrt(x0)
    return [s1*r1, c1*r1, s2*r2, c2*r2]

class CustomQuadEnv(Env):
    metadata = {"render_modes": ["human", "headless"], "render_fps": 30}
    
    def __init__(self,render_mode = None, episodes = 1):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        
        self.episodes = episodes

        self.render_mode=render_mode
        if render_mode == "human":
            self.pybullet_client = bc.BulletClient(connection_mode=p.GUI)
        if render_mode == "headless":
            self.pybullet_client = bc.BulletClient()

        self.plane_id =self.pybullet_client.loadURDF("plane.urdf")
        self.pybullet_client.changeDynamics(self.plane_id, -1, lateralFriction=1.0)

        self.pybullet_client.setGravity(0, 0, -9.81)
        self.pybullet_client.setTimeStep(SIM_TIME)
        urdfFlags = self.pybullet_client.URDF_USE_SELF_COLLISION
        self.quadruped = self.pybullet_client.loadURDF("a1/urdf/a1.urdf", [0, 0, 0.5], [0, 0, 0, 1], flags=urdfFlags, useFixedBase=False)  

        self.observation_space = spaces.Box(low=np.array([-10]*33), high=np.array([10]*33), dtype=np.float32)

        self.action_space = spaces.Box(low=np.array([-1]*12), high=np.array([1]*12), dtype=np.float32)

        self.iteration = 0

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)

        self.cl =(self.iteration)/(2048*150)
        self._control_latency = 0.0
        self.jointIds=[]
        self.paramIds=[]
        self.print_time=time.time()
        self._control_observation = []
        self.last_action = np.zeros(12)

        self.real_joint_values = np.zeros(12)
        
        self.arr_obs = []
        self.pos_obs = np.zeros((12,))
        self.vel_obs = np.zeros((12,))
        self.torque_obs = np.zeros((12,))
        self.base_pos_obs = np.zeros((3,))
        self.base_orientation_obs = np.zeros((4,))
        self.action_obs = np.zeros((12,))
        self.base_ang_vel = np.zeros((3,))
        self.base_lin_vel = np.zeros((3,))
        self.joint_error = np.zeros((12,))
        self.prev_velocity = np.zeros((12,))

        self.pybullet_client.removeAllUserParameters()
        i=0
        self.height_array = []
        self.grav_array = []
        roll = np.random.uniform(-np.pi, np.pi)
        pitch = np.random.uniform(-np.pi/2, np.pi/2)
        yaw = 0  

        # Create a rotation object from Euler angles
        rotation = R.from_euler('xyz', [roll, pitch, yaw])

        # Convert to quaternion
        quaternion = rotation.as_quat()
        self.pybullet_client.resetBasePositionAndOrientation(self.quadruped, [0, 0, 0.5], quaternion)
        for j in range (self.pybullet_client.getNumJoints(self.quadruped)):
            info = self.pybullet_client.getJointInfo(self.quadruped,j)
            jointName = info[1]
            jointType = info[2]
            
            if (jointType==self.pybullet_client.JOINT_PRISMATIC or jointType==self.pybullet_client.JOINT_REVOLUTE):
                
                self.pybullet_client.resetJointState(self.quadruped,j,random.uniform(info[8],info[9]))
                i+=1
                self.jointIds.append(j)
                
        self.pybullet_client.setRealTimeSimulation(False)        
        for _ in range(int((0.75)/SIM_TIME)):    
            self.pybullet_client.stepSimulation()
            time.sleep(SIM_TIME)
        
        self.start_time = time.time()

        self.get_observation()
        return self.arr_obs, {}

    def step(self, action):
        targetPos=[]
        n=0
        
        for indx, i in enumerate(self.jointIds):
            joint_states = self.pybullet_client.getJointState(self.quadruped, i)
            info = self.pybullet_client.getJointInfo(self.quadruped,i)
            targetPos.append(np.interp(action[indx], [-1, 1], [info[8], info[9]]))

            self.pybullet_client.setJointMotorControl2(self.quadruped, i, self.pybullet_client.POSITION_CONTROL,
                                                       targetPosition=targetPos[indx], 
                                                       force=info[10]/3,
                                                       maxVelocity=info[11]/3)
        for _ in range(int((1/80)/SIM_TIME)):    
            self.pybullet_client.stepSimulation()
            if self.render_mode == "human":
                time.sleep(SIM_TIME)

        self.get_observation()
        self.last_action = action
        self.iteration+=1

        done = self.iteration % 1024 == 0

        reward=self.get_reward(action)
        self.prev_velocity=self.joint_velocities
        return self.arr_obs , reward, done, False, {}

    def render(self):
        pass

    def close(self):
        pass

    def parameter_randomizer(self, robot, link_indx):
        link = self.pybullet_client.getDynamicsInfo(robot, link_indx)
        gauss_mass = np.random.normal(link[0], link[0]*0.1)
        gauss_lateral_friction = np.random.normal(link[1], link[1]*0.1)
        gauss_rolling_friction = np.random.normal(link[6], link[6]*0.1)
        self.pybullet_client.changeDynamics(self.quadruped, link_indx , 
                        mass = gauss_mass, 
                        lateralFriction = gauss_lateral_friction, 
                        rollingFriction = gauss_rolling_friction)
    
    
    def GetBasePosition(self):
        _, orientation = (self.pybullet_client.getBasePositionAndOrientation(self.quadruped))
        position = self.pybullet_client.getLinkState(self.quadruped, 0)[0]
        return np.asarray(position, dtype=np.float32), np.asarray(orientation, dtype=np.float32)
    
    def GetBaseVelocity(self):
        baseVel=self.pybullet_client.getBaseVelocity(self.quadruped)
        return np.asarray(baseVel[0]) ,np.asarray(baseVel[1])

        
    def GetJointStates(self):
        joint_positions = np.zeros(12)
        joint_velocities = np.zeros(12)
        joint_torques = np.zeros(12)

        for i,elem in enumerate(self.jointIds):
            joint_states = self.pybullet_client.getJointState(self.quadruped, elem)
            joint_positions[i] = joint_states[0]
            joint_velocities[i] = joint_states[1]
            joint_torques[i] = joint_states[3]
        return joint_positions, joint_velocities, joint_torques
    
    def get_observation(self):
        """
        All of the observation space

        """
        joint_positions, self.joint_velocities, joint_torques = self.GetJointStates()
        self.real_joint_values = joint_positions
        for i in range(len(self.jointIds)):
            info = self.pybullet_client.getJointInfo(self.quadruped, self.jointIds[i])
        
            self.pos_obs[i] = (np.array([np.interp(joint_positions[i], [info[8], info[9]], [-1, 1])],dtype=np.float32))
            self.vel_obs[i] = np.array([np.interp(self.joint_velocities[i], [-info[11]/3, info[11]/3], [-1, 1])],dtype=np.float32)
            
            self.torque_obs[i] = np.array([np.interp(joint_torques[i], [-info[10], info[10]], [-1, 1])],dtype=np.float32)
            self.joint_error[i]= np.array([np.interp(abs(joint_positions[i]-default_pos[i]), [0, max(abs(info[9]-default_pos[i]),abs(info[8]-default_pos[i]))], [1, 0])],dtype=np.float32)

        self.base_pos_obs, self.base_orientation_obs = self.GetBasePosition()
        self.base_lin_vel, self.base_ang_vel= self.GetBaseVelocity()
        self.action_obs = self.last_action.astype("float32")
        r = R.from_quat(self.base_orientation_obs)
        rot_mat = np.linalg.inv(r.as_matrix()).dot(grav_vec)
        self.arr_obs = np.concatenate((self.pos_obs, self.vel_obs, rot_mat, self.base_lin_vel, self.base_ang_vel))
        self.arr_obs = self.arr_obs.astype("float32")

    def map_to_pi(self, angle):
        return (angle+np.pi) % (2*np.pi) - np.pi
    
    def get_reward(self, action):
        """
        Reward function
    
        """
        height_reward = 0
        pos_reward = 0
        vel_reward = 0
        torque_reward = 0
        action_diff = 0
        grav_reward = 0

        
        for i in range(len(self.jointIds)):
            info = self.pybullet_client.getJointInfo(self.quadruped, self.jointIds[i])
            a = (self.real_joint_values[i] - default_pos[i])
            a = self.map_to_pi(a)
            upper = (max(self.map_to_pi(default_pos[i]-info[8]), self.map_to_pi(info[9]-default_pos[i])))
            pos_reward += np.interp(np.abs(a), [0, upper], [0, 1])
            vel_reward += np.abs(self.vel_obs[i])**2/12
            torque_reward += (self.torque_obs[i])**2/12

            action_diff += (abs(action[i] - self.action_obs[i])/2)**2/12
        
        r = R.from_quat(self.base_orientation_obs)
        rot_mat = np.linalg.inv(r.as_matrix()).dot(grav_vec)
        try:
            grav_reward = np.linalg.norm(grav_vec - rot_mat, ord=2)
            grav_reward = np.interp(grav_reward, [0, 2], [1, 0])
        except:
            grav_reward=0
        
        if self.base_pos_obs[2] < 0.31:
            height_reward = (self.base_pos_obs[2]/0.31)
        elif self.base_pos_obs[2] >= 0.31:
            height_reward = 1
        
        bodyflag=self.pybullet_client.getContactPoints(self.quadruped,self.plane_id,0)

        foot_contact_rew=0
        for i in range(4):
            if self.pybullet_client.getContactPoints(self.quadruped,self.plane_id, i*4+5):
                foot_contact_rew+=1

        body_contact_rew=1
        if bodyflag:
            body_contact_rew=0
        
        self.height_array.append(self.base_pos_obs[2])
        self.grav_array.append(1-grav_reward)

        base_vel_reward=gaussian_rbf(self.base_lin_vel,[0]*3,-2)
        base_ang_reward=gaussian_rbf(self.base_ang_vel,[0]*3,-2)
        cl = ((grav_reward))**3
   
        sum_reward = 8*(grav_reward) + 10*height_reward + 2*foot_contact_rew/4 + 2*body_contact_rew + cl*(- 1.9*vel_reward -0.75*torque_reward + 1.5*base_vel_reward + 0.5*base_ang_reward - 1.4*action_diff)

        #if time.time()-self.print_time>1.5:
        #    print(f"Gravity Reward: {grav_reward}")
        #    print(f"Position Reward: {(pos_reward)}")
        #    print(f"Height: {self.base_pos_obs[2]}")
        #    print(f"Total Sum Reward: {sum_reward}\n")
        #     print(f"Vel Reward: {(1 - vel_reward/12)}")
        #     print(f"Torque Reward: {(1 - torque_reward/12)}")
        #     print(f"Base Vel Reward: {  3*base_vel_reward}")
        #     print("\n")
        #     self.print_time=time.time()
        
        return float(sum_reward/1024)


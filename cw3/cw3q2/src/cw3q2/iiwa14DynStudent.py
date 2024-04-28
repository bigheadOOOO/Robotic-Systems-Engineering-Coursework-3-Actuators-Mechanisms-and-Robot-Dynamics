#!/usr/bin/env python3

import numpy as np
from cw3q2.iiwa14DynBase import Iiwa14DynamicBase
from cw3q2.iiwa14DynKDL import Iiwa14DynamicKDL


class Iiwa14DynamicRef(Iiwa14DynamicBase):
    def __init__(self):
        super(Iiwa14DynamicRef, self).__init__(tf_suffix='ref')

    def forward_kinematics(self, joints_readings, up_to_joint=7):
        """This function solve forward kinematics by multiplying frame transformation up until a specified
        joint. Reference Lecture 9 slide 13.
        Args:
            joints_readings (list): the state of the robot joints.
            up_to_joint (int, optional): Specify up to what frame you want to compute forward kinematics.
                Defaults to 7.
        Returns:
            np.ndarray The output is a numpy 4*4 matrix describing the transformation from the 'iiwa_link_0' frame to
            the selected joint frame.
        """

        assert isinstance(joints_readings, list), "joint readings of type " + str(type(joints_readings))
        assert isinstance(up_to_joint, int)

        T = np.identity(4)
        # iiwa base offset
        T[2, 3] = 0.1575

        # 1. Recall the order from lectures. T_rot_z * T_trans * T_rot_x * T_rot_y. You are given the location of each
        # joint with translation_vec, X_alpha, Y_alpha, Z_alpha. Also available are function T_rotationX, T_rotation_Y,
        # T_rotation_Z, T_translation for rotation and translation matrices.
        # 2. Use a for loop to compute the final transformation.
        for i in range(0, up_to_joint):
            T = T.dot(self.T_rotationZ(joints_readings[i]))
            T = T.dot(self.T_translation(self.translation_vec[i, :]))
            T = T.dot(self.T_rotationX(self.X_alpha[i]))
            T = T.dot(self.T_rotationY(self.Y_alpha[i]))

        assert isinstance(T, np.ndarray), "Output wasn't of type ndarray"
        assert T.shape == (4, 4), "Output had wrong dimensions"

        return T

    def get_jacobian_centre_of_mass(self, joint_readings, up_to_joint=7):
        """Given the joint values of the robot, compute the Jacobian matrix at the centre of mass of the link.
        Reference - Lecture 9 slide 14.

        Args:
            joint_readings (list): the state of the robot joints.
            up_to_joint (int, optional): Specify up to what frame you want to compute the Jacobian.
            Defaults to 7.

        Returns:
            jacobian (numpy.ndarray): The output is a numpy 6*7 matrix describing the Jacobian matrix defining at the
            centre of mass of a link.
        """
        assert isinstance(joint_readings, list)
        assert len(joint_readings) == 7
        # 1. Compute the forward kinematics (T matrix) for all the joints.
        # 2. Compute forward kinematics at centre of mass (T_cm) for all the joints. 
        # 3. From the computed forward kinematic and forward kinematic at CoM matrices,
        # extract z, z_cm (z axis of the rotation part of T, T_cm) and o, o_cm (translation part of T, T_cm) for all links.
        # 4. Based on the computed o, o_cm, z, z_cm, fill J_p and J_o matrices up until joint 'up_to_joint'. Apply equations at slide 15, Lecture 9.
        # 5. Fill the remaining part with zeroes and return the Jacobian at CoM.

        # Your code starts here ----------------------------
        # Convert joint readings to KDL JntArray

        jacobian=np.zeros(shape=(6,len(joint_readings)))
        z0=np.zeros (shape=3)
        z0[-1]=1
        tran_o_0=np.zeros(shape=3)
        T0i_joint_COM=self.forward_kinematics_centre_of_mass(joint_readings,up_to_joint)
        pl=T0i_joint_COM[:3,3]
        jacobian[:3,0]=np.cross(z0,(pl-tran_o_0))
        jacobian[3:,0]=z0
        for i in range(1,up_to_joint):
            T_i=self.forward_kinematics(joint_readings,i)

            jacobian[:3,i]=np.cross(T_i[:3,2],(pl-T_i[:3,3]))
            jacobian[3:,i]=T_i[:3,3]

        # 5. Fill the remaining part with zeroes and return the Jacobian at CoM.
        # No need to do this
        # Your code ends here ------------------------------

        assert jacobian.shape == (6, 7)
        return jacobian
    

    def forward_kinematics_centre_of_mass(self, joints_readings, up_to_joint=7):
        """This function computes the forward kinematics up to the centre of mass for the given joint frame.
        Reference - Lecture 9 slide 14.
        Args:
            joints_readings (list): the state of the robot joints.
            up_to_joint (int, optional): Specify up to what frame you want to compute forward kinematicks.
                Defaults to 5.
        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix describing the pose of frame_{up_to_joint} for the
            centre of mass w.r.t the base of the robot.
        """
        T= np.identity(4)
        T[2, 3] = 0.1575

        T = self.forward_kinematics(joints_readings, up_to_joint-1)
        T = T.dot(self.T_rotationZ(joints_readings[up_to_joint-1]))
        T = T.dot(self.T_translation(self.link_cm[up_to_joint-1, :]))

        return T

    def get_B(self, joint_readings):
        """Given the joint positions of the robot, compute inertia matrix B.
        Args:
            joint_readings (list): The positions of the robot joints.

        Returns:
            B (numpy.ndarray): The output is a numpy 7*7 matrix describing the inertia matrix B.
        """
        B = np.zeros((7, 7))
        
        # Some useful steps:
        # 1. Compute the jacobian at the centre of mass from second joint to last joint
        # 2. Compute forward kinematics at centre of mass from second to last joint
        # 3. Extract the J_p and J_o matrices from the Jacobian centre of mass matrices
        # 4. Calculate the inertia tensor using the rotation part of the FK centre of masses you have calculated
        # 5. Apply the the equation from slide 16 lecture 9        
	    # Your code starts here ------------------------------
        # Q5: Can I use code in LAB 9? NO
        # 1. Compute the jacobian at the centre of mass from second joint to last joint
        
        J_cm = np.zeros(shape=(6,7,6))
        # 2. Compute forward kinematics at centre of mass from second to last joint
        T_cm = np.zeros(shape=(4,4,6))
        R_G = np.zeros(shape=(3,3,6))
        I_l = np.zeros(shape=(3,3,6))
        #pl = np.zeros(shape=(3,1,6))
        for i in range(6):
            T_cm[:,:,i] = self.forward_kinematics_centre_of_mass(joint_readings, i+1) # from second joint to last joint
            J_cm[:,:,i] = self.get_jacobian_centre_of_mass(joint_readings, i+1)
            R_G[:,:,i] = T_cm[:3,:3,i]
        # 3. Extract the J_p and J_o matrices from the Jacobian centre of mass matrices
        J_p = J_cm[:3, :, :] # 3*7*6
        J_o = J_cm[3:, :, :] # 3*7*6
        # 4. Calculate the inertia tensor using the rotation part of the FK centre of masses you have calculated
        inertia_tensor = np.zeros(shape=(3,3,6))
        b = np.zeros(shape=(7,7,6))
        
        for i in range(6):
            I_l[0,0,i] = self.Ixyz[i+1,0]
            I_l[1,1,i] = self.Ixyz[i+1,1]
            I_l[2,2,i] = self.Ixyz[i+1,2]
            inertia_tensor[:,:,i] = R_G[:,:,i]@I_l[:,:,i]@R_G[:,:,i].T
            b[:,:,i] = self.mass[i+1]*J_p[:,:,i].T@J_p[:,:,i]+J_o[:,:,i].T@inertia_tensor[:,:,i]@J_o[:,:, i]
            
        # 5. Apply the the equation from slide 16 lecture 9        
        B = np.sum(b,axis=2)
        
        # Your code ends here ------------------------------

        return B

    def get_C_times_qdot(self, joint_readings, joint_velocities):
        """Given the joint positions and velocities of the robot, compute Coriolis terms C.
        Args:
            joint_readings (list): The positions of the robot joints.
            joint_velocities (list): The velocities of the robot joints.

        Returns:
            C (numpy.ndarray): The output is a numpy 7*1 matrix describing the Coriolis terms C times joint velocities.
        """
        assert isinstance(joint_readings, list)
        assert len(joint_readings) == 7
        assert isinstance(joint_velocities, list)
        assert len(joint_velocities) == 7
        # Some useful steps:
        # 1. Create a h_ijk matrix (a matrix containing the Christoffel symbols) and a C matrix.
        # 2. Compute the derivative of B components for the given configuration w.r.t. joint values q. Apply equations at slide 19, Lecture 9.
        # 3. Based on the previously obtained values, fill h_ijk. Apply equations at slide 19, Lecture 9.
        # 4. Based on h_ijk, fill C. Apply equations at slide 19, Lecture 9.

        # Your code starts here ------------------------------  
        h_ijk = np.zeros(shape=(7,7,7))
        C = np.zeros(shape=(7,7))

        def compute_partial_B_partial_q(joint_readings, epsilon=1e-7):
            num_joints = len(joint_readings)
            partial_B_partial_q_k = np.zeros((num_joints, num_joints, num_joints))
            partial_B_partial_q_i = np.zeros((num_joints, num_joints, num_joints))
            B = self.get_B(joint_readings)
            for k in range(num_joints):
                q_plus_eps = joint_readings.copy()
                q_plus_eps[k] += epsilon
                B_plus_eps = self.get_B(q_plus_eps)

                q_min_eps = joint_readings.copy()
                q_min_eps[k] -= epsilon
                B_min_eps = self.get_B(q_min_eps)
                partial_B_partial_q_k[:, :, k] = (B_plus_eps - B_min_eps) / (2*epsilon)

                

            for i in range(num_joints):
                q_plus_eps = joint_readings.copy()
                q_plus_eps[i] += epsilon
                B_plus_eps = self.get_B(q_plus_eps)

                q_min_eps = joint_readings.copy()
                q_min_eps[i] -= epsilon
                B_min_eps = self.get_B(q_min_eps)

                partial_B_partial_q_i[i, :, :] = (B_plus_eps - B_min_eps) / (2*epsilon)

            return partial_B_partial_q_k, partial_B_partial_q_i
        partial_B_partial_q_k, partial_B_partial_q_i = compute_partial_B_partial_q(joint_readings)

        h_ijk = partial_B_partial_q_k - 1/2*partial_B_partial_q_i

        cij = h_ijk@joint_velocities
        

        C = np.sum(cij*joint_velocities, axis=1)
        # Your code ends here ------------------------------
        

        assert isinstance(C, np.ndarray)
        assert C.shape == (7,)
        return C

    def get_G(self, joint_readings):
        """Given the joint positions of the robot, compute the gravity matrix g.
        Args:
            joint_readings (list): The positions of the robot joints.

        Returns:
            g (numpy.ndarray): The output is a numpy 7*1 numpy array describing the gravity matrix g.
        """
        assert isinstance(joint_readings, list)
        assert len(joint_readings) == 7
        # Some useful steps:
        # 1. Compute the Jacobian at CoM for all joints.
        # 2. Use the computed J_cm to fill the G matrix. This method is actually different from what seen in the lectures.
        # 3. Alternatvely, you can compute the P matrix and use it to fill the G matrix based on formulas presented at slides 17 and 22, Lecture 9.
        # Your code starts here ------------------------------
        # 1. Compute the Jacobian at CoM for all joints.
        # J_cm = np.zeros(shape=(6,7,7))
        def get_P(joint_readings): 
            assert len(joint_readings) == 7
            p_i = np.zeros(shape=(7,))
            pl = np.zeros(shape=(3,len(joint_readings)))
            g0T = np.array([0,0,-self.g])
            # T0i_joint_COM = np.identity(4)
            for i in range(7):
                # J_cm[:,:,i] = self.get_jacobian_centre_of_mass(joint_readings, i)
                pl[:, i] = self.forward_kinematics_centre_of_mass(joint_readings, i+1)[:3, -1]
                p_i[i] = -self.mass[i]*(g0T@pl[:,i])
                
            P = np.sum(p_i) # 1x7, 1x3, 3x7 => 7x7 => 7*1
            print('P', P)
            return P
        # 2. Use the computed J_cm to fill the G matrix. This method is actually different from what seen in the lectures.
        def compute_partial_P_partial_q(joint_readings, epsilon=1e-8):
            P = get_P(joint_readings)
            partial_P_partial_q = np.zeros(shape=(7,))
            for i in range(len(joint_readings)):
                q_plus_eps = joint_readings.copy()
                q_plus_eps[i] += epsilon
                P_plus_eps = get_P(q_plus_eps)

                q_min_eps = joint_readings.copy()
                q_min_eps[i] -= epsilon
                P_min_eps = get_P(q_min_eps)

                partial_P_partial_q[i] = (P_plus_eps - P_min_eps) / (2*epsilon)
            assert (partial_P_partial_q.shape == (7,))
            return partial_P_partial_q
        g = compute_partial_P_partial_q(joint_readings)
        # Your code ends here ------------------------------

        assert isinstance(g, np.ndarray)
        assert g.shape == (7,)
        return g
    
    @staticmethod
    def kdl_jnt_array_to_list(kdl_array):
        joints = []
        for i in range(0, 7):
            joints.append(kdl_array[i])
        return joints


#!/usr/bin/env python3
##TODO: Implement your cw3q5 coding solution here.

import numpy as np
import rospy
import rosbag
import rospkg
import PyKDL
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cw3q2.iiwa14DynKDL import Iiwa14DynamicKDL
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker
import rospy
from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt

class iiwa14Accelerations(object):
    def __init__(self):
        # Initialize node
        rospy.init_node('iiwa14_cw3', anonymous=True)
        self.current_joint_torque = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.all_accelarations = []
        self.joint_names = ["iiwa_joint_1", "iiwa_joint_2", "iiwa_joint_3", "iiwa_joint_4", "iiwa_joint_5", "iiwa_joint_6", "iiwa_joint_7"]
        self.time_stamps = []
        self.end_flag = 1
        self.target_joint_duration_secs=[]
        # Save question number for check in main run method
        self.kdl_iiwa14 = Iiwa14DynamicKDL()

        # Create trajectory publisher and a checkpoint publisher to visualize checkpoints
        self.traj_pub = rospy.Publisher('/iiwa/EffortJointInterface_trajectory_controller/command', JointTrajectory,
                                              queue_size=5)
        self.checkpoint_pub = rospy.Publisher("checkpoint_positions", Marker, queue_size=100)
    def run(self):
        """This function is the main run function of the class. When called, it runs question 6 by calling the q6()
        function to get the trajectory. Then, the message is filled out and published to the /command topic.
        """
        print("run q6a")
        rospy.loginfo("Waiting 1 second for everything to load up.")
        rospy.sleep(1.0)
        traj = self.q5()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = self.joint_names
        self.traj_pub.publish(traj)

    def q5(self):
        """ This is the main q6 function. Here, other methods are called to create the shortest path required for this
        question. Below, a general step-by-step is given as to how to solve the problem.
        Returns:
            traj (JointTrajectory): A list of JointTrajectory points giving the robot joint positions to achieve in a
            given time period.
        """
        # 1. Load in targets from the bagfile (checkpoint data and target joint positions).
        target_cart_tf, target_joint_positions, target_joint_velocities, target_joint_accelerations, self.target_joint_duration_secs, target_joint_duration_nsecs = self.load_targets()
        # 2. Create a JointTrajectory message.
        # Create a trajectory message and publish to get the robot to move to this checkpoints
        # Call the publish_checkpoints function to publish the found Cartesian positions of the loaded joints
        self.publish_traj_tfs(target_cart_tf)

        traj = JointTrajectory()
        t = 0
        for i in range(target_joint_positions.shape[1]):
            traj_point = JointTrajectoryPoint()
            traj_point.positions = target_joint_positions[:, i]
            if i > 0:
                t = t + self.target_joint_duration_secs[i] - self.target_joint_duration_secs[i - 1]
            traj_point.time_from_start.secs = t
            traj_point.time_from_start.nsecs = target_joint_duration_nsecs[i]
            traj_point.velocities = target_joint_velocities[:,i]
            traj_point.accelerations = target_joint_accelerations[:,i]

            traj.points.append(traj_point)

        assert isinstance(traj, JointTrajectory)
        return traj
    def joint_torque_callback(self, msg):
        """ ROS callback function for joint states of the robot. Broadcasts the current pose of end effector.

        Args:
            msg (JointState): Joint state message containing current robot joint position.

        """
        self.current_joint_torque = self.kdl_iiwa14.kdl_jnt_array_to_list(msg.effort)
        self.cal_acceleration(self.current_joint_torque)
    def cal_acceleration(self, torques):
        B = self.kdl_iiwa14.get_B(self.kdl_iiwa14.current_joint_position) # 7x7
        Cqdot = np.array(self.kdl_iiwa14.get_C_times_qdot(self.kdl_iiwa14.current_joint_position, self.kdl_iiwa14.current_joint_velocity)) # 7x1
        g = np.array(self.kdl_iiwa14.get_G(self.kdl_iiwa14.current_joint_position)) # 7x1
        # accelerations = np.linalg.inv(B)@((np.array(torques)).reshape(7,1) - Cqdot - g)
    
        accelerations = np.linalg.inv(B)@((np.array(torques) - Cqdot - g).reshape(7,1))
        # print('accelerations:', rospy.get_time(), len(self.time_stamps), len(self.all_accelarations))
        current_time = rospy.get_time()
        self.time_stamps.append(current_time)
        self.all_accelarations.append(accelerations)
    def load_targets(self):# at current time, I assume 7 joints and 1 movements. The first one descirbe the initial position.  
        """This function loads the checkpoint data from the 'data.bag' file. In the bag file, you will find messages
        relating to the target joint positions. You need to use forward kinematics to get the goal end-effector position.
        Returns:
            target_cart_tf (4x4x5 np.ndarray): The target 4x4 homogenous transformations of the checkpoints found in the
            bag file. There are a total of 5 transforms (4 checkpoints + 1 initial starting cartesian position).
            target_joint_positions (5x5 np.ndarray): The target joint values for the 4 checkpoints + 1 initial starting
            position.
        """
        # Defining ros package path
        rospack = rospkg.RosPack()
        path = rospack.get_path('cw3q5')

        # Initialize arrays for checkpoint transformations and joint positions
        target_joint_positions = np.zeros(shape=(7, 4))
        target_joint_velocities = np.zeros(shape=(7, 4))
        target_joint_accelerations = np.zeros(shape=(7, 4))
        target_joint_duration_secs = []
        target_joint_duration_nsecs = []
        # Create a 4x4 transformation matrix, then stack 6 of these matrices together for each checkpoint
        target_cart_tf = np.repeat(np.identity(4), 4, axis=1).reshape((4, 4, 4))

        # Load path for selected question
        bag = rosbag.Bag(path + '/bag/cw3q5.bag')
        # Get the current starting position of the robot
        target_joint_positions[:, 0] = self.kdl_iiwa14.kdl_jnt_array_to_list(self.kdl_iiwa14.current_joint_position)
        target_joint_velocities[:, 0] = self.kdl_iiwa14.kdl_jnt_array_to_list(self.kdl_iiwa14.current_joint_velocity)
        target_joint_duration_secs.append(0)
        target_joint_duration_nsecs.append(0)
        # Initialize the first checkpoint as the current end effector position

        target_cart_tf[:, :, 0] = self.kdl_iiwa14.forward_kinematics(list(self.kdl_iiwa14.current_joint_position))
        i = 1
        for topic, msg, t in bag.read_messages(topics=['/iiwa/EffortJointInterface_trajectory_controller/command']):
            # get joint position
            for j in range(len(msg.points)): 

                joint_positions = self.kdl_iiwa14.kdl_jnt_array_to_list(msg.points[j].positions)
                joint_velocities = self.kdl_iiwa14.kdl_jnt_array_to_list(msg.points[j].velocities)
                joint_accelerations = self.kdl_iiwa14.kdl_jnt_array_to_list(msg.points[j].accelerations)
                joint_duration_secs = msg.points[j].time_from_start.secs
                joint_duration_nsecs = msg.points[j].time_from_start.nsecs

                # save joint position to array
                target_joint_positions[:,j+1] = joint_positions
                target_joint_velocities[:,j+1] = joint_velocities
                target_joint_accelerations[:,j+1] = joint_accelerations
                target_joint_duration_secs.append(joint_duration_secs)
                target_joint_duration_nsecs.append(joint_duration_nsecs)
                
                # save tf to array
                target_cart_tf[:,:,j+1] = self.kdl_iiwa14.forward_kinematics(list(msg.points[j].positions))
            i += 1
            print('target_joint_duration_secs', target_joint_duration_secs)
        # Close the bag
        bag.close()

        assert isinstance(target_cart_tf, np.ndarray)
        assert target_cart_tf.shape == (4, 4, 4)
        assert isinstance(target_joint_positions, np.ndarray)
        assert target_joint_positions.shape == (7,4)

        return target_cart_tf, target_joint_positions, target_joint_velocities, target_joint_accelerations, target_joint_duration_secs, target_joint_duration_nsecs

    
    def publish_traj_tfs(self, tfs):
        """This function gets a np.ndarray of transforms and publishes them in a color coded fashion to show how the
        Cartesian path of the robot end-effector.
        Args:
            tfs (np.ndarray): A array of 4x4xn homogenous transformations specifying the end-effector trajectory.
        """
        id = 0
        for i in range(0, tfs.shape[2]):
            marker = Marker()
            marker.id = id
            id += 1
            marker.header.frame_id = 'world'
            marker.header.stamp = rospy.Time.now()
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0 + id * 0.05
            marker.color.b = 1.0 - id * 0.05
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = tfs[0, -1, i]
            marker.pose.position.y = tfs[1, -1, i]
            marker.pose.position.z = tfs[2, -1, i]
            self.checkpoint_pub.publish(marker)


    def get_jacobian(self, joint):
        """Compute the jacobian matrix using KDL library.

        Args:
            joint (numpy.array): NumPy array of size 5 corresponding to number of joints on the robot.

        Returns:
            Jacobian (numpy.ndarray): NumPy matrix of size 6x5.

        """
        joints_kdl = self.kdl_iiwa14.list_to_kdl_jnt_array(joint)
        jac_kdl = PyKDL.Jacobian(self.kdl_iiwa14.kine_chain.getNrOfJoints())
        self.kdl_iiwa14.jac_calc.JntToJac(joints_kdl, jac_kdl)
        jac = self.kdl_iiwa14.kdl_to_mat(jac_kdl)
        return jac
    

if __name__ == '__main__':
    try:
        iiwa14planner = iiwa14Accelerations()
        
        iiwa14planner.run()
        joint_torque_sub = rospy.Subscriber('/iiwa/joint_states', JointState, iiwa14planner.joint_torque_callback,
                                                queue_size=5)
        start_t = rospy.get_time()
        while (not rospy.is_shutdown()) and rospy.get_time() < start_t+iiwa14planner.target_joint_duration_secs[-1]+5:
            if iiwa14planner.all_accelarations:
                plt.clf()
                all_accelarations = np.squeeze(np.array(iiwa14planner.all_accelarations))
                # Assuming joint names are available in joint_state.name
                for i, joint_name in enumerate(iiwa14planner.joint_names):
                    plt.plot(iiwa14planner.time_stamps[:all_accelarations.shape[0]], [acc[i] for acc in all_accelarations], label=joint_name)

                plt.xlabel('Time (s)')
                plt.ylabel('Joint Acceleration')
                plt.legend()
                plt.pause(0.01)  # Adjust the pause time as needed for real-time updates

            rospy.sleep(0.1)  # Adjust the sleep time based on your application needs
        print('end')
        if rospy.get_time() >= start_t+iiwa14planner.target_joint_duration_secs[-1]+5:
            rospy.signal_shutdown("Shutdown initiated")
        # plt.ioff()
        plt.show()
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass
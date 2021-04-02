#include "std_msgs/String.h"
#include "sensor_msgs/JointState.h"
#include <math.h>
#include <abb_libegm/egm_controller_interface.h>
#include <abb_libegm/egm_trajectory_interface.h>
#include <iostream>
#include <vector>
#include "robot_state_publisher/robot_state_publisher.h"
#include "robot_state_publisher/joint_state_listener.h"
#include <boost/asio.hpp>

class Connect
{
private:
	ros::NodeHandle nh_;
    boost::asio::io_service io_service;
    boost::thread_group thread_group;
    abb::egm::BaseConfiguration configuration;
    abb::egm::wrapper::Output zero_output;
    sensor_msgs::JointState joint_states;
    ros::Publisher jts_pub;
    ros::Publisher joint_state_pub;
    ros::Subscriber sim_sub;
public:
    abb::egm::wrapper::Input  input;
    abb::egm::wrapper::Output output;
    abb::egm::wrapper::Joints initial_velocity;
    //abb::egm::wrapper::Joints initial_positions;
    bool wait, finished; 
    int first_visit = 1;
    int sim_button_presses = 0;
    std::vector<double> velocity {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; 
    std::vector<double> position {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  
    Connect(short int port_num);
    void check_connection(abb::egm::EGMControllerInterface &EGM_IF, bool &wait);
    void make_connection();
    //void callback(sensor_msgs::JointState js, abb::egm::EGMControllerInterface &EGM_IF);
    void callback(sensor_msgs::JointState js);
    void sim_callback(std_msgs::String msg);
    void write_state(abb::egm::EGMControllerInterface &interface, abb::egm::wrapper::Output &output, abb::egm::wrapper::Joints &initial_velocity);
    void write_zero(abb::egm::EGMControllerInterface &interface, abb::egm::wrapper::Output &output);
    void publish_joint_states();
};

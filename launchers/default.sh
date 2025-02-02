#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

export ROSCONSOLE_STDOUT_LINE_BUFFERED=1

echo $ROS_MASTER_URI
echo $ROS_HOSTNAME
export ROS_HOSTNAME=localhost

source /code/catkin_ws/devel/setup.bash --extend
source /code/submission_ws/devel/setup.bash --extend
source /code/solution/devel/setup.bash --extend

dt-exec roslaunch --wait agent agent_node.launch &
dt-exec roslaunch --wait car_interface all.launch veh:=$VEHICLE_NAME &
dt-exec roslaunch --wait object_detection_custom object_detection_node.launch veh:=$VEHICLE_NAME

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join

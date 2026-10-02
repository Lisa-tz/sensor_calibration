import csv
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


class ImuRecorder(Node):
    """Logs sensor_msgs/Imu gyro + accel to CSV (rad/s, m/s^2)."""

    def __init__(self):
        super().__init__('imu_recorder')
        self.declare_parameter('topic', '/imu/data_raw')
        self.declare_parameter('output', 'imu_log.csv')
        self.declare_parameter('duration', 600.0)

        topic = self.get_parameter('topic').value
        out = os.path.expanduser(self.get_parameter('output').value)
        self.duration = float(self.get_parameter('duration').value)

        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        self.file = open(out, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['t', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])

        self.t0 = None
        self.count = 0
        self.done = False
        self.create_subscription(Imu, topic, self.cb, qos_profile_sensor_data)
        self.create_timer(5.0, self.report)
        self.get_logger().info(f'Recording {topic} -> {out} for {self.duration:.0f} s')

    def cb(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if t == 0.0:  # driver didn't stamp: use receive time
            t = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = t
        t -= self.t0
        w, a = m.angular_velocity, m.linear_acceleration
        self.writer.writerow([f'{t:.6f}', w.x, w.y, w.z, a.x, a.y, a.z])
        self.count += 1
        if t >= self.duration:
            self.done = True

    def report(self):
        self.get_logger().info(f'{self.count} samples logged')

    def close(self):
        self.file.close()
        self.get_logger().info(f'Saved {self.count} samples')


def main():
    rclpy.init()
    node = ImuRecorder()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

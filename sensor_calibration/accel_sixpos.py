import json
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

# (label, axis index, sign of the gravity reading expected on that axis)
POSES = [('+X', 0, 1), ('-X', 0, -1), ('+Y', 1, 1), ('-Y', 1, -1), ('+Z', 2, 1), ('-Z', 2, -1)]


class SixPosRecorder(Node):
    """Guided six-position accelerometer capture. Output units: m/s^2."""

    def __init__(self):
        super().__init__('accel_sixpos')
        self.declare_parameter('topic', '/imu/data_raw')
        self.declare_parameter('output', 'accel_sixpos.json')
        self.declare_parameter('settle', 3.0)      # s to wait after pressing Enter
        self.declare_parameter('duration', 15.0)   # s of data per pose
        p = self.get_parameter
        self.output = os.path.expanduser(p('output').value)
        self.settle = float(p('settle').value)
        self.duration = float(p('duration').value)
        self.buf = []
        self.collecting = False
        self.create_subscription(Imu, p('topic').value, self.cb, qos_profile_sensor_data)

    def cb(self, m):
        if self.collecting:
            a = m.linear_acceleration
            self.buf.append((a.x, a.y, a.z))

    def capture(self):
        time.sleep(self.settle)
        self.buf = []
        self.collecting = True
        time.sleep(self.duration)
        self.collecting = False
        return np.array(self.buf)


def main():
    rclpy.init()
    node = SixPosRecorder()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    poses = {}
    try:
        for label, axis, sign in POSES:
            while True:
                input(f'\nPose {label}: put the sensor so its {label} axis points UP '
                      '(the opposite face rests on the table).\n'
                      'Keep it perfectly still, then press Enter... ')
                print(f'  settling {node.settle:.0f} s, then recording {node.duration:.0f} s: do not touch')
                data = node.capture()
                if len(data) < 10:
                    print('  No data received. Is the driver running? Try again.')
                    continue
                mean, std = data.mean(axis=0), data.std(axis=0)
                half = len(data) // 2
                drift = np.abs(data[:half].mean(axis=0) - data[half:].mean(axis=0)).max()
                print(f'  n={len(data)}  mean={np.round(mean, 4).tolist()}  std={np.round(std, 4).tolist()}')
                problems = []
                if int(np.argmax(np.abs(mean))) != axis or np.sign(mean[axis]) != sign:
                    problems.append(f'reading does not look like {label} up')
                if std.max() > 0.1:
                    problems.append(f'noisy (std {std.max():.3f} m/s^2): vibration?')
                if drift > 0.05:
                    problems.append(f'it moved or drifted while recording ({drift:.3f} m/s^2)')
                if problems:
                    for pr in problems:
                        print('  WARNING:', pr)
                    if input('  Redo this pose? [Y/n] ').strip().lower() != 'n':
                        continue
                poses[label] = {'mean': mean.tolist(), 'std': std.tolist(), 'n': int(len(data))}
                break
    except (KeyboardInterrupt, EOFError):
        print('\nAborted.')
    finally:
        if len(poses) == 6:
            os.makedirs(os.path.dirname(os.path.abspath(node.output)), exist_ok=True)
            with open(node.output, 'w') as f:
                json.dump({'units': 'm/s^2', 'poses': poses}, f, indent=2)
            print(f'\nSaved {node.output}')
        else:
            print('\nIncomplete, nothing saved.')
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

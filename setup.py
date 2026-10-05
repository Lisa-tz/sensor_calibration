from setuptools import setup, find_packages

package_name = 'sensor_calibration'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lisa',
    maintainer_email='lisa@example.com',
    description='Record and analyze IMU/sensor calibration data',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'imu_recorder = sensor_calibration.imu_recorder:main',
            'analyze_gyro = sensor_calibration.analyze_gyro:main',
            'accel_sixpos = sensor_calibration.accel_sixpos:main',
            'analyze_accel = sensor_calibration.analyze_accel:main',
        ],
    },
)

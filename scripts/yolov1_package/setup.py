from setuptools import setup, find_packages

setup(
    name='yolov1_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.0.0',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'train-yolo=train:main',
        ],
    },
)

from setuptools import setup, find_packages

setup(
    name="handheld_scanner",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pyrealsense2",
        "open3d",
        "opencv-python",
        "pyyaml",
    ],
)

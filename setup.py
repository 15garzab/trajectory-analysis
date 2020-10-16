from setuptools import setup, find_packages

setup(
    name='trajectory-analysis',
    version='0.1.0',
    packages=find_packages(include=['traj_standalone', 'traj_importer']),
    install_requires=['numpy', 'pandas', 'matplotlib', 'seaborn', 'ovito'],
)

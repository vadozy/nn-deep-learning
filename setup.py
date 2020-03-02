from setuptools import setup
from setuptools import find_packages

setup(name='nn-deep-learning',
      version='0.1',
      description='Neural Networks and Deep Learning, p3 src for http://neuralnetworksanddeeplearning.com/',
      url='https://github.com/vadozy/nn-deep_learning',
      install_requires=['numpy', 'scikit-learn', 'scipy', 'Theano'],
      license='MIT',

      packages=find_packages(where='src'),
      package_dir={'': 'src'},

      zip_safe=False)

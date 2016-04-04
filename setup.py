from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='rnndatasets',
      version='0.1',
      description='Some datasets, mostly set up for use with Tensorflow.',
      url='https://github.com/PFCM/datasets',
      author='pfcm',
      packages=['funniest'],
      license='BSD 3-clause',
      zip_safe=False,
      install_requires=[
        'numpy',
        'six'
      ])

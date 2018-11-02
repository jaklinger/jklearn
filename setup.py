from setuptools import setup
from setuptools import find_namespace_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

exclude = ['docs', 'tests*']
common_kwargs = dict(
    version='0.1',
    license='MIT',
    install_requires=required,
    long_description=open('README.rst').read(),
    url='https://github.com/jaklinger/jklearn',
    author='Joel Klinger',
    author_email='joel.klinger@nesta.org.uk',
    maintainer='Joel Klinger',
    maintainer_email='joel.klinger@nesta.org.uk',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Environment :: MacOS X'
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>3.6',
    include_package_data=True,
)

setup(name='jklearn',
      packages=find_namespace_packages(where='.', exclude=exclude),
      **common_kwargs)

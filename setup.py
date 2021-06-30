import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='tralda',
    version='0.0.1',
    author='David Schaller',
    author_email='sdavid@bioinf.uni-leipzig.de',
    description='A library for tree data structures and algorithms.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/david-schaller/tralda',
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.5',
    install_requires=[
        'numpy>=1.16.4',
        'networkx>=2.2',
   ],
)

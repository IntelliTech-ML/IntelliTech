from setuptools import setup, find_packages

setup(
    name='IntelliNeuro',
    version='0.1.0',
    author='Ajay Soni',
    description='Perceptron implementation for linear, binary, and multi-class tasks',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ml-beginner-learner/IntelliNode',  # optional but helpful
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn'
    ]
)

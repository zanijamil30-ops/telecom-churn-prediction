from setuptools import setup, find_packages

setup(
    name='telecom-churn-prediction',
    version='1.0.0',
    author='zanjijamil30-ops',
    author_email='',
    description='Machine Learning project for predicting telecom customer churn using ensemble and stacking models.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/zanjijamil30-ops/telecom-churn-prediction',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'scipy',
        'matplotlib',
        'ipywidgets'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)


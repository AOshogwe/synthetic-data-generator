from setuptools import setup, find_packages

setup(
    name="synthetic_data_generator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "sqlalchemy",
        "matplotlib",
        "seaborn",
    ],
    entry_points={
        'console_scripts': [
            'synthetize=synthetic_data_generator.cli.main:main',
        ],
    },
)
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="loan_credit_default_mlops",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    description="A machine learning project for predicting loan credit default.",
    author="Salma Ahmed"
)
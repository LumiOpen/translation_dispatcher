from setuptools import setup, find_packages

setup(
    name="dispatcher",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "requests",
    ],
    extras_require={
        "dev": [
            "pytest",
            "responses",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            "dispatcher-server=dispatcher.server:main",
        ],
    },
)

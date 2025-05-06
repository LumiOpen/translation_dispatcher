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
            # starts the FastAPI/uvicorn server
            "dispatcher-server = dispatcher.server:main",

            # generic task-runner CLI
            "dispatcher-task-run = dispatcher.taskmanager.cli:main",
        ],
    },
)

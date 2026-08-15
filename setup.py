from setuptools import setup, find_packages

def read_readme():
    with open("README.md", "r") as fh:
        return fh.read()

def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="MemHawk",
    version="1.0.0",
    author="Arvid Bouziane",
    author_email="arvid.bouziane@icloud.com",
    packages=find_packages(),
    description="MemHawk is a universal program that stores past chat turns in a vector database and retrieves only the most relevant memories, so the AI receives a much smaller context window without losing important information.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Hawk3388/MemHawk",
    license='Apache 2.0',
    python_requires='>=3.10',
    install_requires=read_requirements()
)
"""
Setup configuration for Luma Memory Module.
"""

from setuptools import setup, find_packages

setup(
    name="luma-memory",
    version="0.1.0",
    author="Luma Team",
    author_email="team@luma.ai",
    description="Central memory system for Luma personal AI",
    long_description="The Luma Memory Module provides persistent storage and retrieval of user actions and context summaries for the Luma personal AI system.",
    long_description_content_type="text/plain",
    url="https://github.com/luma/luma-memory",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "sqlalchemy>=2.0.23",
        "cryptography>=41.0.7",
        "pydantic>=2.10.0",
        "pydantic-settings>=2.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "httpx>=0.25.2",
            "hypothesis>=6.92.0",

        ],

    },
    entry_points={
        "console_scripts": [
            "luma-memory-server=luma_memory.api:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

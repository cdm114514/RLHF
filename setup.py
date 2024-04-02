import setuptools

setuptools.setup(
    name="RLHF",  # Replace with your own package name
    version="0.1.0",  # Package version
    author="cdm",  # Replace with your name
    author_email="3493665571@qq.com",  # Replace with your email
    url="https://github.com/cdm114514/RLHF",  # Package website or source code URL
    packages=setuptools.find_packages(),  # Automatically find package directories
    classifiers=[  # Classifiers help users find your project
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "datasets>=2.8.0", "sentencepiece>=0.1.97", "protobuf==3.20.3",
        "accelerate>=0.15.0", "torch>=1.12.0",
        "transformers>=4.31.0,!=4.33.2", "tensorboard","fairscale==0.4.13"
    ]
)

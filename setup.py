from setuptools import setup, find_packages

setup(
    name="xwrl6432-adc-reader",
    version="0.1.0",
    description="Real-time ADC reader for xWRL6432 mmWave radar via DCA1000",
    author="Leon Braungardt",
    package_dir={'': 'src'},
    packages=find_packages(where='src', exclude=["examples", "radar_config", "images"]),
    install_requires=[
        "numpy",
        "pyserial",
        "tqdm"
    ],
    python_requires=">=3.10",
)
import os

this_directory = os.path.abspath(os.path.dirname(__file__))


def run_setup():
    # fixes warning https://github.com/pypa/setuptools/issues/2230
    from setuptools import setup, find_packages

    setup(
        name="MINE",
        author="Wojciech Kretowicz",
        version="0.1.0",
        install_requires=[
            'numpy>=1.18.4',
            'pandas>=1.1.2',
            'tqdm'
        ],
        packages=find_packages(include=["MINE", "MINE.*"]),
        python_requires='>=3.7',
        include_package_data=True
    )


if __name__ == "__main__":
    run_setup()
    # pdoc command: pdoc --html attackpdp --force

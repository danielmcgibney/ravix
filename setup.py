from setuptools import setup, find_packages
from setuptools.command.install import install
import os
from ravix._version import __version__ as VERSION

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        print(r"""
 ____                   
|  _ \ __ ___   _(_)_  __
| |_) / _` \ \ / / \ \/ /
|  _ < (_| |\ V /| |>  < 
|_| \_\__,_| \_/ |_/_/\_\
                   
      Ravix: Applied modeling and visualization for business analytics
        """)
        print(f"Regress v{VERSION} successfully installed!")
        print("Documentation: https://businessregression.com\n")

# Attempt to read the README.md file for the long description
readme_path = 'README.md'
if os.path.exists(readme_path):
    with open(readme_path, 'r') as fh:
        long_description = fh.read()
else:
    long_description = 'Long description not available.'

setup(
    name='ravix',
    version=VERSION,
    packages=find_packages(include=['ravix', 'ravix.*']),
    install_requires=[
        'matplotlib', 'pandas', 'numpy', 'statsmodels', 'seaborn', 'scikit-learn', 
        'scipy'
    ],
    cmdclass={'install': PostInstallCommand},
    entry_points={
        'console_scripts': [
            # Define any command-line scripts here, if applicable
        ],
    },
    author="Daniel McGibney",
    author_email="dmcgibney@bus.miami.edu",
    description="Applied modeling and visualization for business analytics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danmcgib/ravix",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
    package_data={
        'ravix': ['data/*.csv'],
    },
)

#import setuptools
from setuptools import setup, Extension
from distutils.util import convert_path


setup(
    name='didip_util',
    version='0.0.2',
    packages=['ddp_util'],
    #package_dir={'ddp_util': 'src/ddp_util'},
    package_dir={'':'src'},
    #package_data={'frat': ['resources/*.js', 'resources/*.json', 'resources/*.jinja2']},
    #include_package_data=True,
    scripts=['bin/ddp_leech_sheets', 'bin/ddp_leech_monasterium', 'bin/ddp_leech_charter'],
    license='GPLv3',
    author='Anguelos Nicolaou et al.',
    author_email='anguelos.nicolaou@gmail.com',
    url='https://zimlab.uni-graz.at/gams/projects/didip/general',
    description="DiDip management codebase",
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    #download_url='https://github.com/anguelos/tormentor/archive/0.1.0.tar.gz',
    keywords=["documents", "diplomatics", "monasterium"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering"],
    install_requires=["tqdm", "bs4", "lxml", "python-magic", "fargv"],
)
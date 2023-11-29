from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = '''One package open your gate to using different type of GAN'''

# Setting up
setup(
    name="magic",
    version=VERSION,
    author="Sushanta Das",
    author_email="<imachi@skiff.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy','matplotlib','click','scikit-learn','pandas','tqdm','joblib','torch','torchvision'],
    keywords=['python', 'GAN','Generative adversarial networks','scikit-learn', 'machine learning', 'deep learning', 'Computer Vision', 'Artificial intelligence'],
)
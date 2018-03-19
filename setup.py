from setuptools import setup

DESCRIPTION = "Detection of Online Impersonation based on User Profiling in Social Networks through Optimized SVM"
LONG_DESCRIPTION = DESCRIPTION
NAME = "osvm"
AUTHOR = "Christel Junco, Charlotte Sampiano, Chester Sumampong"
AUTHOR_EMAIL = "christeltheresejunco@gmail.com,charlotte_sampiano@umindanao.edu.ph"
MAINTAINER = "Christel Junco, Charlotte Sampiano"
MAINTAINER_EMAIL = "christeltheresejunco@gmail.com,charlotte_sampiano@umindanao.edu.ph"
DOWNLOAD_URL = 'https://github.com/ChristelJunco/THESIS-Junco-Sampiano'
LICENSE = 'MIT'

VERSION = '1.3.0'

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=DOWNLOAD_URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['osvm'],
      package_dir={'osvm': 'osvm'},
      install_requires=['matplotlib','numpy','pandas'],
      classifiers=['Development Status :: 4 - Beta',\
                       'Programming Language :: Python :: 3.6',\
                       'License :: OSI Approved :: MIT License',\
                       'Operating System :: OS Independent',\
                       'Intended Audience :: Science/Research',\
                       'Topic :: Scientific/Engineering :: Visualization']
     )
 
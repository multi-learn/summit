# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2025
# -----------------
#
#
# * Université d'Aix Marseille (AMU) -
# * Centre National de la Recherche Scientifique (CNRS) -
# * Université de Toulon (UTLN).
# * Copyright © 2019-2025 AMU, CNRS, UTLN
#
# Contributors:
# ------------
#
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
#
#
# Description:
# -----------
#
# Supervised MultiModal Integration Tool's Readme
# This project aims to be an easy-to-use solution to run a prior benchmark on a dataset and evaluate mono- & multi-view algorithms capacity to classify it correctly.
#
# Version:
# -------
#
# * summit-multi-learn version = 0.0.2
#
# Licence:
# -------
#
# License: New BSD License : BSD-3-Clause
#
#
# ######### COPYRIGHT #########
#
#
#
#Extracting requrements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
import os, re
import shutil
from distutils.command.clean import clean as _clean
from distutils.dir_util import remove_tree
from distutils.command.sdist import sdist
from setuptools import setup, find_packages

USE_COPYRIGHT = True
try:
    from copyright import writeStamp, eraseStamp
except ImportError:
    USE_COPYRIGHT = False

# --------------------------------------------------------------------
# Clean target redefinition - force clean everything supprimer de la liste '^core\.*$',
relist = ['^.*~$', '^#.*#$', '^.*\.aux$', '^.*\.pyc$', '^.*\.o$']
reclean = []

###################
# Get Summit version
####################
def get_version():
    v_text = open('VERSION').read().strip()
    v_text_formted = '{"' + v_text.replace('\n', '","').replace(':', '":"')
    v_text_formted += '"}'
    v_dict = eval(v_text_formted)
    return v_dict["summit"]

########################
# Set Summit __version__
########################
def set_version(summit_dir, version):
    filename = os.path.join(summit_dir, '__init__.py')
    buf = ""
    for line in open(filename, "rb"):
        if not line.decode("utf8").startswith("__version__ ="):
            buf += line.decode("utf8")
    f = open(filename, "wb")
    f.write(buf.encode("utf8"))
    f.write(('__version__ = "%s"\n' % version).encode("utf8"))

for restring in relist:
    reclean.append(re.compile(restring))


def wselect(args, dirname, names):
    for n in names:
        for rev in reclean:
            if (rev.match(n)):
                os.remove("%s/%s" %(dirname, n))
        break

class clean(_clean):
    def walkAndClean(self):
        os.walk("..", wselect, [])
        pass

    def run(self):
        clean.run(self)
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('multimodal'):
            for filename in filenames:
                if (filename.endswith('.so') or
                        filename.endswith('.pyd') or
                        filename.endswith('.dll') or
                        filename.endswith('.pyc')):
                    os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))

##############################
# Custom sdist command
##############################
class m_sdist(sdist):
    """ Build source package

    WARNING : The stamping must be done on an default utf8 machine !
    """
    def run(self):
        if USE_COPYRIGHT:
            writeStamp()
            sdist.run(self)
            # eraseStamp()
        else:
            sdist.run(self)

# Ceci n'est qu'un appel de fonction. Mais il est trèèèèèèèèèèès long
# et il comporte beaucoup de paramètres
def setup_package():
    group = 'multi-learn'
    name = 'summit' + group
    version = get_version()
    summit_dir = 'summit'
    set_version(summit_dir, version)
    here = os.path.abspath(os.path.dirname(__file__))
    # Or 'README.rst', depending on your format
    long_description_content_type = 'text/x-rst'
    with open(os.path.join(here, 'README.rst'), encoding='utf-8') as readme:
        long_description = readme.read()

    #url = 'https://github.com/{}/{}'.format(group, name)
    #project_urls = {
    #    'Documentation': 'https://{}.github.io/{}/'.format(group, name),
    #   'Source': url,
    #    'Tracker': '{}/issues'.format(url)}
    packages = find_packages(exclude=['*.test'])
    extras_require = {
         'test' : ['pytest', 'pytest-cov'],
         'doc' : ['sphinx >= 3.0.2', 'numpydoc', 'docutils', 'sphinx-autoapi',
                 'sphinx_rtd_theme']}


    setup(version=version,
    packages=packages,
    long_description=long_description,
    # Active la prise en compte du fichier MANIFEST.in
    include_package_data=True,
    license="BSD-3-Clause",
    license_files="LICENSE",
    extras_require=extras_require
    )

if __name__ == "__main__":
    setup_package()

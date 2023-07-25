# Copyright 2023 The Oxpecker team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages
import os 

# A trick from https://github.com/jina-ai/jina/blob/79b302c93b01689e82cf4b52f46522eb7497c404/setup.py#L20
pkg_name = 'oxpecker'
libinfo_py = os.path.join(pkg_name, '__init__.py')
libinfo_content = open(libinfo_py, 'r', encoding='utf8').readlines()
version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][0]
exec(version_line)  # gives __version__

setup(name         = "oxpecker",
      version      = __version__,
      author       = "Herman Sugi Harto and oxpecker contributor",
      author_email = "hermansh.id@gmail.com",
      license      = "Apache-2.0",
      url          = "https://github.com/hermansh-id/oxpecker",
      package_dir  = {"": pkg_name},
      packages     = find_packages(pkg_name),
      description  = "A document analysis and extractor",
      long_description=open("README.md", "r", encoding="utf-8").read(),
      long_description_content_type="text/markdown",
      python_requires='>=3.6',
      install_requires=[
        "pdf2image",
      ],
      extras_require={
      },
      include_package_data=True
      )
"""Provides a package for handling datasets consistently, could be helpful.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import shutil

from six.moves import xrange
from six.moves import urllib


def maybe_download(filepath, url):
    """Checks to see if a file exists. If not, downloads it from the given url.

    Args:
      filepath (str): the path to check.
      url (str): the url which we will attempt to retrieve if the file is not
        present.

    Returns:
      str: the path to the file
    """
    # let's not use tensorflow for this
    # so that it is a bit more flexible
    if not os.path.exists(filepath):
        # dl into a tempfile in case it faoils
        with tempfile.NamedTemporaryFile() as tmpfile:
            temp_file_name = tmpfile.name
            urllib.request.urlretrieve(url, temp_file_name)
            shutil.copy(temp_file_name, filepath)
            print('Downloaded {}'.format(filepath))
    return filepath

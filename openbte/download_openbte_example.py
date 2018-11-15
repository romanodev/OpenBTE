
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from past.builtins import basestring
from builtins import object
from past.utils import old_div
import urllib.request, urllib.parse, urllib.error


def main():

 urllib.request.urlretrieve('https://www.dropbox.com/s/adaeafjnzipitu0/example.py?dl=1','example.py')

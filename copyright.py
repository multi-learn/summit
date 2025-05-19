# -*- coding: utf-8 -*-
from __future__ import print_function, division
import time
import os
import sys
import fileinput


def findFiles(directory, files=[]):
    """scan a directory for py, pyx, pxd extension files."""
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path) and (path.endswith(".py") or
                                     path.endswith(".pyx") or
                                     path.endswith(".pxd")):
            if filename != "__init__.py" and filename != "version.py":
                files.append(path)
        elif os.path.isdir(path):
            findFiles(path, files)
    return files


def fileUnStamping(filename):
    """ Remove stamp from a file """
    is_stamp = False
    for line in fileinput.input(filename, inplace=1):
        if line.find("# COPYRIGHT #") != -1:
            is_stamp = not is_stamp
        elif not is_stamp:
            print(line, end="")


def fileStamping(filename, stamp):
    """ Write a stamp on a file

    WARNING : The stamping must be done on an default utf8 machine !
    """
    old_stamp = False  # If a copyright already exist over write it.
    for line in fileinput.input(filename, inplace=1):
        if line.find("# COPYRIGHT #") != -1:
            old_stamp = not old_stamp
        elif line.startswith("# -*- coding: utf-8 -*-"):
            print(line, end="")
            print(stamp)
        elif not old_stamp:
            print(line, end="")


def getStamp(date, summit_version):
    """ Return the corrected formated stamp """
    stamp = open("copyrightstamp.txt").read()
    stamp = stamp.replace("DATE", date)
    stamp = stamp.replace("SUMMIT_VERSION", summit_version)
    stamp = stamp.replace('\n', '\n# ')
    stamp = "# " + stamp
    stamp = stamp.replace("# \n", "#\n")
    return stamp.strip()


def getVersionsAndDate():
    """ Return (date, summit_version..
    ) """
    v_text = open('VERSION').read().strip()
    v_text_formted = '{"' + v_text.replace('\n', '","').replace(':', '":"')
    v_text_formted += '"}'
    v_dict = eval(v_text_formted)
    return (time.strftime("%Y"), v_dict['summit'])


def writeStamp():
    """ Write a copyright stamp on all files """
    stamp = getStamp(*getVersionsAndDate())
    files = findFiles(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "summit"))
    for filename in files:
        fileStamping(filename, stamp)
    fileStamping("setup.py", stamp)
    fileStamping("format_dataset.py", stamp)


def eraseStamp():
    """ Erase a copyright stamp from all files """
    files = findFiles(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "summit"))
    for filename in files:
        fileUnStamping(filename)
    fileUnStamping("setup.py")
    fileUnStamping("format_dataset.py")


def usage(arg):
    print("Usage :")
    print("\tpython %s stamping" % arg)
    print("\tpython %s unstamping" % arg)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        usage(sys.argv[0])
    elif len(sys.argv) == 2:
        if sys.argv[1].startswith("unstamping"):
            eraseStamp()
        elif sys.argv[1].startswith("stamping"):
            writeStamp()
        else:
            usage(sys.argv[0])
    else:
        usage(sys.argv[0])

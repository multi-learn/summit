import errno
import os


def secure_file_path(file_name):  # pragma: no cover
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

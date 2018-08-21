import os


def safe_mkdir(path):
    """Create a directory if there isn't one already.

    Args:
        path: string, directory
    """
    if not os.path.exists(path):
        os.mkdir(path)

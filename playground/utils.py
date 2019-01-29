import sys
import termios


def wait_for_key():
    # From: https://stackoverflow.com/a/34956791

    result = None
    fd = sys.stdin.fileno()

    old_term = termios.tcgetattr(fd)
    new_attr = termios.tcgetattr(fd)
    new_attr[3] = new_attr[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, new_attr)

    try:
        result = sys.stdin.read(1)
    except IOError:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, old_term)

    return result

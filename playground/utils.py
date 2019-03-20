import sys
import termios
from datetime import datetime


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


def test_env_fps(env, steps=10_000):
    done = True
    env.seed(367)
    a = env.action_space.sample() * 0.0
    tic = datetime.now()
    for i in range(steps):
        if done:
            env.reset()
        done = env.step(a.copy())[2]
    total_time_s = (datetime.now() - tic).total_seconds()
    fps = steps / total_time_s
    print(f'FPS: {fps}')

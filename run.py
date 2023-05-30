from utils.logger import test_logger

import os


def main():
    test_logger.info('Test logger hello! 3 env\n')

    for k, v in os.environ.items():
        test_logger.info(f"Env: {k}:{v}")

    test_logger.info('\nTest logger bye! 3 env\nrunning attack manager:\n')

    # exec(open('manager_example.py').read())
    os.system('python3 -m unittest discover')

    test_logger.info('\nDone')


if __name__ == '__main__':
    main()
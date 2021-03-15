from TraderEnv import OhlcvEnv

ENV_NAME = 'OHLCV-v0'
TIME_STEP = 30
print('ok')

# Get the environment and extract the number of actions.
PATH_TRAIN = "./data/train/"
PATH_TEST = "./data/test/"
env = OhlcvEnv(TIME_STEP, path=PATH_TRAIN)


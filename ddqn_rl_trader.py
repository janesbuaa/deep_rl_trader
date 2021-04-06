import numpy as np
import os
import process_data
import pandas as pd
from pathlib import Path
# import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, CuDNNLSTM, Dropout
from keras.optimizers import Adam

# keras-rl agent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from sklearn import preprocessing

# trader environment
from TraderEnv import OhlcvEnv


def create_model(shape, nb_actions):
    model = Sequential()
    model.add(CuDNNLSTM(64, input_shape=shape, return_sequences=True))
    model.add(CuDNNLSTM(64))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


def load_from_csv(path):
    file_list = [x.name for x in Path(path).iterdir() if x.is_file()]
    file_list.sort()
    rand_episode = file_list.pop()
    raw_df = pd.read_csv(path + rand_episode)
    extractor = process_data.FeatureExtractor(raw_df)
    df = extractor.add_bar_features()  # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features
    df = extractor.add_mv_avg_features()
    df = extractor.add_adj_features()
    df = extractor.add_ta_features()

    df.dropna(inplace=True)  # drops Nan rows
    df = df.iloc[:, 4:].astype('float32')
    df = preprocessing.StandardScaler().fit(df).transform(df)
    return df


def main():
    # OPTIONS
    ENV_NAME = 'OHLCV-v0'
    TIME_STEP = 300
    BATCH_SIZE = 30
    # 1000-100=1s 1000-250=2.2s,1000-500=4.18s,500-500=2.1s,500-250=1.1s

    # Get the environment and extract the number of actions.
    PATH_TRAIN = "./data/train/"
    # PATH_TEST = "./data/test/"
    PATH_MODEL = "./model/duel_dqn_weights.h5f"
    df = load_from_csv(PATH_TRAIN)
    tran = int(len(df)*0.8)

    env = OhlcvEnv(TIME_STEP, df[:tran, :])
    env_test = OhlcvEnv(TIME_STEP, df[tran+1:, :], train=False)

    # random seed
    np.random.seed(BATCH_SIZE + TIME_STEP)
    env.seed(BATCH_SIZE + TIME_STEP)

    nb_actions = env.action_space.n
    model = create_model(shape=env.shape, nb_actions=nb_actions)
    # print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and even the metrics!
    memory = SequentialMemory(limit=20000, window_length=TIME_STEP)
    # policy = BoltzmannQPolicy()
    policy = EpsGreedyQPolicy()
    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, batch_size=BATCH_SIZE,
                   nb_steps_warmup=int(BATCH_SIZE + TIME_STEP),
                   enable_dueling_network=True, dueling_type='avg', target_model_update=100, policy=policy, gamma=.95,
                   processor=None)
    dqn.compile(Adam(lr=1e-2), metrics=['mae'])

    if os.path.exists(PATH_MODEL):
        dqn.load_weights(PATH_MODEL)
        print('Load weights success!')

    ite = 0
    while True:
        dqn.fit(env, nb_steps=20000, nb_max_episode_steps=20000, visualize=False, verbose=2)
        ite += 1
        try:
            if ite >= 40 and ite % 10 == 0:
                info = dqn.test(env_test, nb_episodes=1, visualize=False)
                n_long, n_short, total_reward, portfolio = info['n_trades']['long'], info['n_trades']['short'], info['total_reward'], int(info['portfolio'])
                np.array([info]).dump('./info/duel_dqn_{0}_weights_{1}LS_{2}_{3}_{4}.info'.format(ENV_NAME, portfolio, n_long, n_short, total_reward))
                dqn.save_weights('./model/duel_dqn_{0}_weights_{1}LS_{2}_{3}_{4}.h5f'.format(ENV_NAME, portfolio, n_long, n_short, total_reward), overwrite=True)
            else:
                dqn.save_weights('./model/duel_dqn_weights.h5f', overwrite=True)
        except KeyboardInterrupt:
            continue


if __name__ == '__main__':
    main()

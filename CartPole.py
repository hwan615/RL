import gym
import random
import numpy as np

env = gym.make('CartPole-v1', render_mode="human")
goal_steps = 500
# observation, info = env.reset()

# for _ in range(goal_steps):
#     observation, reward, terminated, truncated, info = env.step(random.randrange(0,2))
#     print(observation)
#     print(reward)
#     if terminated or truncated:
#         observation, info = env.reset()
# env.close()

def data_preparation(N, K, f, render=False):
    game_data=[]
    for i in range(N):
        score = 0
        game_steps = []
        observation, info = env.reset()
        ## observation [카트 위치, 카트 속도, 막대의 각도, 끝에서의 막대의 속도]
        print(observation)
        for step in range(goal_steps):
            ## action 0,1 오른쪽 혹은 왼쪽으로의 방향 결정
            action = f(observation)
            ## action 을 취했을 때의 observation 과 action 을 game_step 에 저장
            game_steps.append((observation, action))
            ## 다음 action 정보
            observation, reward, terminated, truncated, info = env.step(action)
            ## 현재 액션에 대한 보상
            score += reward
            if terminated or truncated:
                break
        ## action 이 실패하면 진행된 observation, action 과 score 정보를 데이터에 담는다
        game_data.append((score, game_steps))
    ## game_data 를 score 내림차순으로 정렬
    game_data.sort(key=lambda s:-s[0])

    training_set = []
    ## game_data 에서 상위 K 개의 데이터에 대한 정보 저장
    for i in range(K):
        for step in game_data[i][1]:
            if step[1] == 0:
                training_set.append((step[0], [1, 0]))
            else:
                training_set.append((step[0], [0, 1]))
    print("{0}/{1}th score: {2}".format(K, N, game_data[K-1][0]))
    return training_set 

training_data = data_preparation(10, 5, lambda s: random.randrange(0,2))

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def build_model():
    model = Sequential()
    model.add(Dense(128, input_dim=4, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='mse', optimizer=Adam())
    return model

def train_model(model, training_set):
    X = np.array([i[0] for i in training_set]).reshape(-1, 4)
    y = np.array([i[1] for i in training_set]).reshape(-1, 2)
    model.fit(X, y, epochs=10)

if __name__ == '__main__':
    N = 1000
    K = 50
    model = build_model()
    training_data = data_preparation(N, K, lambda s: random.randrange(0,2))
    train_model(model, training_data)

    def predictor(s):
        return np.random.choice([0,1], p=model.predict(s, reshape(-1, 4))[0])

    data_preparation(100, 100, predictor, True)

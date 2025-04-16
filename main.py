import GameEnv
import pygame
import numpy as np
from ddqn_keras import DDQNAgent
from tensorflow.keras.models import load_model #type: ignore

TOTAL_GAMETIME = 5000
N_EPISODES = 1000
REPLACE_TARGET = 50

game = GameEnv.RacingEnv()
game.fps = 60

ddqn_agent = DDQNAgent(
    alpha=0.001,             # Learning rate for the optimizer (how big each parameter update step is)
    gamma=0.99,               # Discount factor for future rewards (how much future rewards matter)
    n_actions=5,              # Number of possible actions the agent can take (like [left, right, accelerate, brake, do nothing])
    epsilon=1.00,             # Initial value for epsilon in epsilon-greedy policy (chance to explore random actions)
    epsilon_end=0.10,         # Minimum value for epsilon after decay (so agent keeps some randomness)
    epsilon_dec=0.9999,       # Epsilon decay factor after each step or episode (controls how quickly it shifts from explore to exploit)
    replace_target=REPLACE_TARGET,  # Number of episodes or steps after which to update the target network
    batch_size=512,           # Number of experiences sampled from memory during each learning step
    input_dims=19             # Size of the input state vector (number of features per observation from the environment)
)


# ddqn_agent.load_model()

ddqn_scores = []
eps_history = []


def run():
    best_score = -np.inf  # Initialize best score

    for e in range(N_EPISODES):
        game.reset()
        done = False
        score = 0
        counter = 0
        gtime = 0

        observation_, reward, done = game.step(0)
        observation = np.array(observation_, dtype=np.float32)

        renderFlag = (e % 10 == 0 and e > 0)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            action = ddqn_agent.choose_action(observation)
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_, dtype=np.float32)

            # print(observation_)
            # print(type(observation_))

            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward

            ddqn_agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            ddqn_agent.learn()

            gtime += 1
            if gtime >= TOTAL_GAMETIME:
                done = True

            if renderFlag:
                game.render(action)

        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[max(0, e-100):(e+1)])

        if e % REPLACE_TARGET == 0 and e > REPLACE_TARGET:
            ddqn_agent.update_network_parameters()

        if e % 10 == 0 and e > 10:
            ddqn_agent.save_model()
            print("Model is being saved...")

        # Saving the best model
        if score > best_score:
            best_score = score
            ddqn_agent.brain_eval.model.save("best_ddqn_model.keras")
            print(f'🎉 New Best Score: {best_score:.2f}. Best model saved as best_model.keras!', flush=True)


        print(f'Episode: {e}, Score: {score:.2f}, Average Score: {avg_score:.2f}, '
              f'Epsilon: {ddqn_agent.epsilon:.3f}, Memory Size: {ddqn_agent.memory.mem_cntr % ddqn_agent.memory.mem_size}',flush=True)

def run_best_model():
    ddqn_agent.brain_eval.model = load_model("best_ddqn_model.keras")
    ddqn_agent.epsilon = 0.0  # Fully greedy
    ddqn_agent.brain_target.copy_weights(ddqn_agent.brain_eval)  # Sync target network
    print("✔️ Loaded best_model.keras for demo run.")

    game.reset()
    done = False
    score = 0
    gtime = 0

    observation_, reward, done = game.step(0)
    observation = np.array(observation_)

    # 🔍 Test what action values the loaded model outputs on first observation
    test_action_values = ddqn_agent.brain_eval.predict(np.array([observation]))
    print("Test action values (Q-values) for initial state:", test_action_values)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        action = ddqn_agent.choose_action(observation)
        print(f"Action: {action}, Score: {score:.2f}")

        observation_, reward, done = game.step(action)
        observation_ = np.array(observation_)

        score += reward
        observation = observation_

        gtime += 1
        if gtime >= TOTAL_GAMETIME:
            done = True

        game.render(action)

    print(f"▶️ Final Demo Score: {score:.2f}", flush=True)



if __name__ == "__main__":
    run()  # Train

    # Run the best model
    # run_best_model()
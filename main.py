import env
import ddpg


def main():
    v = env.Env()
    episodes = 200
    agent = ddpg.DeepDeterministicPolicyGradient(episodes)
    pool = env.ExperiencePool()
    real_epoch = 0
    for episode in range(episodes):
        done = False
        state = v.observe()
        epoch = 0
        while not done:
            print(f'Episode: {episode}, Epoch: {epoch}, position: {state}')
            action = agent([state], batch_size=1, train=True, episode=episode)
            go = v.choose(action)
            reward, done = v.run(go)
            nstate = v.observe()
            pool.append(env.Experience(
                state=state,
                action=action,
                reward=reward,
                nstate=nstate,
                done=done,
            ))
            state = nstate
            if len(pool) > 100:
                minibatch = pool.sample()
                agent.train_networks(minibatch, real_epoch)
            real_epoch += 1
            epoch += 1
        v.reset()

    # Evaluate
    total_reward = []
    for episode in range(episodes):
        done = False
        epoch = 0
        state = v.observe()
        episode_reward = 0
        while not done:
            print(f'Episode: {episode}, Epoch: {epoch}, position: {state}')
            action = agent([state], batch_size=1, train=False, episode=episode)
            go = v.choose(action)
            reward, done = v.run(go)
            state = v.observe()
            episode_reward += reward
        total_reward.append(episode_reward)
        v.reset()
    print(total_reward, sum(total_reward))

main()

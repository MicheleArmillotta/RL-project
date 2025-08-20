import numpy as np
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from PPO_clip_shared import Agent
from PPO_clip_shared import criticNet, critic_Train




if __name__ == '__main__':
    print("Inizializzazione dell'ambiente...")
    layout_name = "cramped_room"
    print(f"Layout scelto: {layout_name}")
    LOAD_MODELS = True 
    N_SAVE = 100

    base_mdp = OvercookedGridworld.from_layout_name(layout_name,old_dynamics = True)
    base_env = OvercookedEnv.from_mdp(base_mdp, info_level=0, horizon=400)
    env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
    grid = base_mdp.terrain_mtx
    shared_critic = criticNet(input_dim=env.observation_space.shape)

    N = 1024
    batch_size = 128
    n_epochs = 6
    alpha = 5e-4
    shared_train = critic_Train(env.observation_space.shape,shared_critic,n_epochs,learning_rate=alpha,gamma = 0.99, GAElambda = 0.95, critic_smoother=0.5, eps = 0.2)
    agent0 = Agent(shared_critic,action_dim=env.action_space.n, batch_size=batch_size, 
                    learning_rate=alpha, epochs=n_epochs, 
                    input_dim=env.observation_space.shape, gamma = 0.99, GAElambda=0.95, epsilon=0.2,entropy_coeff=0.01,agent_name="agent0")
    
    agent1 = Agent(shared_critic,action_dim=env.action_space.n, batch_size=batch_size, 
                    learning_rate=alpha, epochs=n_epochs, 
                    input_dim=env.observation_space.shape, gamma = 0.99, GAElambda=0.95, epsilon=0.2,entropy_coeff=0.01,agent_name="agent1")
    n_games = 3000

    best_score = 0
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    visualizer = StateVisualizer()

    start_episode = 0
    if LOAD_MODELS:
        print("\nCARICAMENTO MODELLI")
        
        # Carica critic condiviso
        critic_info = shared_train.load_models()
        
        # Carica agenti
        agent0_info = agent0.load_models()
        agent1_info = agent1.load_models()
        
        # Determina da quale episodio iniziare
        if agent0_info and agent1_info and critic_info:
            start_episode = max(
                agent0_info.get('episode', 0),
                agent1_info.get('episode', 0),
                critic_info.get('episode', 0)
            )
            print(f"Riprendo training dall'episodio: {start_episode}")
        
        print("=== FINE CARICAMENTO ===\n")

    for i in range(n_games):
        observation = env.reset()
        observation0 = observation["both_agent_obs"][0]
        observation1 = observation["both_agent_obs"][1]
        done = False
        score = 0
        tot_shp0 = 0
        tot_shp1 = 0
        while not done:
            #img_path = os.path.join("imgs", f"frame_{n_steps}.png")

            action0, old_prob0 = agent0.generate_action(observation0)
            S_value0 = agent0.predict_value(observation0)
            action1, old_prob1= agent1.generate_action(observation1)
            S_value1 = agent1.predict_value(observation1)

            joint_action = (action0, action1)
            observation_, reward, done, info = env.step(joint_action)
            state_to_render = env.render()
            grid = base_mdp.terrain_mtx

            #visualizer.display_rendered_state(observation_["overcooked_state"],grid = grid,img_path=img_path)
            n_steps += 1

            shp_reward0 = info["shaped_r_by_agent"][0]
            shp_reward1 = info["shaped_r_by_agent"][1]
            tot_shp0 += shp_reward0
            tot_shp1 += shp_reward1

            score += reward 
            agent0.remember(observation0, action0, (reward+shp_reward0),S_value0,old_prob0,done)
            agent1.remember(observation1, action1, (reward+shp_reward1),S_value1,old_prob1,done)
            if n_steps % N == 0:

                if not done:
                    S_value_next0 = agent0.predict_value(observation_["both_agent_obs"][0])
                    agent0.remember_nextS(S_value_next0)
                    S_value_next1 = agent1.predict_value(observation_["both_agent_obs"][1])
                    agent1.remember_nextS(S_value_next1)

                else:
                    agent0.remember_nextS(0) 
                    agent1.remember_nextS(0)
                

                GAE_adv0, GAE_adv1 = shared_train.train_critic(agent0.memory,agent1.memory)

                agent0.train(GAE_adv0)
                agent1.train(GAE_adv1)
                learn_iters += 1
            observation = observation_
            observation0 = observation["both_agent_obs"][0]
            observation1 = observation["both_agent_obs"][1]
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            #agent.save_models()
        if i % 100 == 0:
            print(f"Episode {i}: Recent average shaped rewards - Agent0: {np.mean([tot_shp0]):.2f}, "
            f"Agent1: {np.mean([tot_shp1]):.2f}")
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
            'time_steps', n_steps, 'learning_steps', learn_iters,'reward agent 0 %.1f' % tot_shp0,
            'reward agent 1 %.1f' % tot_shp1)
        if i%N_SAVE == 0:
            agent0.save_models(i,score)
            agent1.save_models(i,score)
            shared_train.save_models(i,score)
    x = [i+1 for i in range(len(score_history))]




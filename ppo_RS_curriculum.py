import numpy as np
import random
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from PPO_clip import Agent
import os
from overcooked_ai_py.mdp.overcooked_mdp import SoupState

class GeneralizedOvercooked:
    def __init__(self ,info_level=0, horizon=400):
        self.layouts=["cramped_room","asymmetric_advantages","coordination_ring",
                                     "forced_coordination"]
        self.prob_last_idx = 0.65
        self.envs = []
        self.base_mdps = []
        self.curriculum = ["cramped_room"]
        self.curr_idx = 0
        self.layout_episode = 0
        for layout in self.layouts:
            base_mdp = OvercookedGridworld.from_layout_name(layout,old_dynamics = True)
            base_env = OvercookedEnv.from_mdp(base_mdp, info_level=info_level, horizon=horizon)
            env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
            self.envs.append(env)
            self.base_mdps.append(base_mdp)
        self.cur_env = self.envs[0]
        self.cur_mdp = self.base_mdps[0]
        self.observation_space, self.action_space = self.cur_env.observation_space, self.cur_env.action_space
    
    def add_layout(self, avg_score):
        if avg_score >= 80 and self.layout_episode >= 300 and len(self.curriculum) < 4:
            self.curr_idx = self.curr_idx + 1
            self.curriculum.append(self.layouts[self.curr_idx])
            self.layout_episode = 0
            print("LAYOUT ADDED:", self.layouts[self.curr_idx])
    
    def reset(self, avg_score):
        if len(self.curriculum) >= 4 and avg_score >= 80 and self.layout_episode >= 500: #true random testing
            idx = random.randint(0, len(self.envs) - 1 )
        else:
            if random.random() < self.prob_last_idx:
                idx = len(self.curriculum) - 1  # 65% 
            else:
                if len(self.curriculum) > 1:
                    idx = random.randint(0, len(self.curriculum) - 2)  # 35% 
                else:
                    idx = 0
        self.cur_env = self.envs[idx]
        self.cur_mdp = self.base_mdps[idx]
        self.layout_episode = self.layout_episode + 1
        print("layout:", self.layouts[idx])
        return self.cur_env.reset()
    
    def step(self, *args):
        return self.cur_env.step(*args)
    
    def render(self, *args):
        return self.cur_env.render(*args)
    
    def get_base_mdp(self):
        return self.cur_mdp
    
    def save(self, path="saved_models/curriculum.txt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for layout in self.curriculum:
                f.write(layout + "\n")
        print(f"Curriculum salvato in: {path}")

    def load(self, path="saved_models/curriculum.txt"):
        if os.path.exists(path):

            with open(path, "r") as f:
                self.curriculum = [line.strip() for line in f.readlines()]
            
            self.curr_idx = len(self.curriculum) - 1
            self.layout_episode = 0

            self.envs = []
            self.base_mdps = []
            for layout in self.layouts:
                base_mdp = OvercookedGridworld.from_layout_name(layout, old_dynamics=True)
                base_env = OvercookedEnv.from_mdp(base_mdp, info_level=0, horizon=400)
                env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
                self.envs.append(env)
                self.base_mdps.append(base_mdp)

            self.cur_env = self.envs[self.layouts.index(self.curriculum[-1])]
            self.cur_mdp = self.base_mdps[self.layouts.index(self.curriculum[-1])]
            self.observation_space = self.cur_env.observation_space
            self.action_space = self.cur_env.action_space

            print(f"Curriculum caricato da: {path}")
        else:
            print(f"nessun file trovato in: {path}")


def shaped_reward(prev_state, next_state,agent_idx,n_steps):
    reward = 0
    #decay_steps = 400000
    p_play = prev_state["overcooked_state"].players[agent_idx]
    n_play = next_state["overcooked_state"].players[agent_idx]

    p_state = prev_state["overcooked_state"]
    n_state = next_state["overcooked_state"]

    #reward shaping

    # + 1 onion = 0.5
    # + 1 onion in the pot = 1
    # soup is ready = 2
    # pick up the ready soup = 1
    # delivering the soup (default) = 20
    # idle = 0

    if not p_play.has_object() and n_play.has_object():
        if n_play.get_object().name == "onion":
            reward += 0.1
            #print("DEBUG = > PRESO UNA CIPOLLA ALLO STEP: ", n_steps)

    if not p_play.has_object() and n_play.has_object():
        if n_play.get_object().name == "dish":
            reward += 0.1
            #print("DEBUG = > PRESO un piatto ALLO STEP: ", n_steps)
        
    def check_soup_taken_simple(prev_state, curr_state):
            prev_soups = {pos: obj for pos, obj in prev_state.objects.items() if obj.name == "soup"}
            curr_soups = {pos: obj for pos, obj in curr_state.objects.items() if obj.name == "soup"}
            
            
            for pos in prev_soups:
                prev_soup = prev_soups[pos]
                
                # Se la zuppa era pronta e ora non c'è più nulla in quella posizione
                if (hasattr(prev_soup, '_cooking_tick') and 
                    prev_soup._cooking_tick >= 20 and 
                    pos not in curr_soups):
                    
                    #print(f"DEBUG => Zuppa pronta scomparsa dalla posizione {pos} - PRESA!")
                    return True
            
            return False

    #    # Utilizzo:
    result = check_soup_taken_simple(p_state, n_state)
    if result:
            reward += 0.4
            print(f"DEBUG => ZUPPA PRESA CONFERMATA ALLO STEP: {n_steps}")


    # Idle penalty
    if (p_play.position == n_play.position and
        ((not p_play.has_object() and not n_play.has_object()) or 
         (p_play.has_object() and n_play.has_object() and 
          p_play.get_object().name == n_play.get_object().name))):
        reward -= 0.001

    #soup is ready

    def soup_ready(state):
        flag = 0
        for state_obj in state.objects.values():
            #for obj in state_obj:
                #print("DEBUG => oggetti ->", state_obj.name)
                if isinstance(state_obj, SoupState):
                    if state_obj.is_ready:
                        flag += 1

        return flag
    

    p_flag = soup_ready(p_state)
    n_flag = soup_ready(n_state)

    if n_flag > p_flag:
        reward += 0.3
        print("DEBUG = > ZUPPA PRONTA ALLO STEP: ", n_steps)

    def plus_one_onion(state):
        ingr = 0
        for state_obj in state.objects.values():
            #for obj in state_obj:
                if isinstance(state_obj, SoupState):
                    ingr += len(state_obj.ingredients)


        return ingr
    
    p_ingr = plus_one_onion(p_state)
    n_ingr = plus_one_onion(n_state)


    if n_ingr > p_ingr:
        reward += 0.2
        print("DEBUG = > CIPOLLA INSERITA NELLA ZUPPA ALLO STEP: ", n_steps)

    
    #shaping_scale = max(1.0 - n_steps / decay_steps, 0.0)
    return reward



if __name__ == '__main__':

    print("Inizializzazione dell'ambiente...")

    env = GeneralizedOvercooked()
    avg_score = 0


    N = 1024
    batch_size = 128
    n_epochs = 6
    alpha = 5e-4
    agent0 = Agent(action_dim=env.action_space.n, batch_size=batch_size, 
                    learning_rate=alpha, epochs=n_epochs, 
                    input_dim=env.observation_space.shape, gamma = 0.99, GAElambda=0.95, epsilon=0.2,entropy_coeff=0.01,critic_smoother= 0.5,eps = 0.2,agent_name="Agent0")
    
    agent1 = Agent(action_dim=env.action_space.n, batch_size=batch_size, 
                    learning_rate=alpha, epochs=n_epochs, 
                    input_dim=env.observation_space.shape, gamma = 0.99, GAElambda=0.95, epsilon=0.2,entropy_coeff=0.01, critic_smoother=0.5, eps = 0.2,agent_name="Agent1")
    n_games = 20000 
    best_score = 0
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    visualizer = StateVisualizer()
    SAVE_MODEL = True
    SAVE_FREQ = 100
    LOAD_MODELS = True

    start_episode = 0
    if LOAD_MODELS:
        print("\nCARICAMENTO MODELLI")
        
        
        agent0_info = agent0.load_models()
        agent1_info = agent1.load_models()
        

        if agent0_info and agent1_info:
            start_episode = max(
                agent0_info.get('episode', 0),
                agent1_info.get('episode', 0)
            )
            print(f"Riprendo training dall'episodio: {start_episode}")
        
        print("CARICAMENTO CURRICULUM")
        
        env.load()
        
        print(" FINE CARICAMENTO \n")

    for i in range(n_games):

        env.add_layout(avg_score)
        observation = env.reset(avg_score)
    

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
            #visualizer.display_rendered_state(observation_["overcooked_state"],grid = grid,img_path=img_path)
            n_steps += 1

            #shp_reward0 = shaped_reward(observation, observation_, 0, n_steps)
            #shp_reward1 = shaped_reward(observation, observation_, 1,n_steps)
            shp_reward0 = info["shaped_r_by_agent"][0]
            shp_reward1 = info["shaped_r_by_agent"][1]
            tot_shp0 += shp_reward0
            tot_shp1 += shp_reward1

            score += reward 
            agent0.remember(observation0, action0, (reward+shp_reward0),S_value0,old_prob0,done)
            agent1.remember(observation1, action1, (reward + shp_reward1),S_value1,old_prob1,done)
            if n_steps % N == 0:
                S_value_next0 = agent0.predict_value(observation_["both_agent_obs"][0])
                agent0.remember_nextS(S_value_next0)
                agent0.train()
                S_value_next1 = agent1.predict_value(observation_["both_agent_obs"][1])
                agent1.remember_nextS(S_value_next1)
                agent1.train()
                learn_iters += 1
            observation = observation_
            observation0 = observation["both_agent_obs"][0]
            observation1 = observation["both_agent_obs"][1]
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            #agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters, "reward_agent1 %.1f" % tot_shp0, 
                "reward_agent2 %.1f" % tot_shp1)
        
        if SAVE_MODEL and (i % SAVE_FREQ == 0):
            agent0.save_models(i,score)
            agent1.save_models(i,score)
            env.save()

    x = [i+1 for i in range(len(score_history))]
    #plot_learning_curve(x, score_history, figure_file)



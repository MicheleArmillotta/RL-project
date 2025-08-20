import os
import numpy as np
import tensorflow as tf
from keras import layers
import tensorflow_probability as tfp
import json

class Memory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.S_values = []
        self.old_probs = []
        self.dones = []
        self.nextS = 0
        self.batch_size = batch_size


    def add_obs(self, state, action, reward, S_value, old_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.S_values.append(S_value)
        self.old_probs.append(old_prob)
        self.dones.append(done)

    def add_nextS(self, nextS):
        self.nextS = nextS

    def clean_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.S_values = []
        self.old_probs = []
        self.dones = []

    def return_batch(self, shuffle = True):  #shuffled minibatch
        data_size = len(self.states)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)

        batches = []
        for start in range(0, data_size, self.batch_size):
            end = start + self.batch_size
            batches.append(indices[start:end])

        return batches
    


    def return_np_arrays(self):
        return np.array(self.states), np.array(self.actions),np.array(self.rewards).flatten(),\
        np.array(self.S_values).flatten(),np.array(self.old_probs),np.array(self.dones).flatten(), self.nextS
    
#must try also with pooling layers (could reduce training time/overfitting)
class actorNet(tf.keras.Model):
    def __init__(self, num_actions):
        super(actorNet, self).__init__()
        self.conv1 = layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')
        self.conv3 = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.1)
        self.logits = layers.Dense(num_actions) 
    
    def call(self, x, training=True):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        if training:
            x = self.dropout(x, training=training)
        return self.logits(x)

class criticNet(tf.keras.Model):
    def __init__(self):
        super(criticNet, self).__init__()
        self.conv1 = layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')
        self.conv3 = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.1)
        self.value = layers.Dense(1) 

    def call(self, x, training=True):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        if training:
            x = self.dropout(x, training=training)
        return self.value(x)
    


class Agent:
    def __init__(self, batch_size, input_dim, action_dim, epochs, gamma, GAElambda, epsilon, learning_rate,entropy_coeff, critic_smoother, eps ,agent_name = "Agent"):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.epochs = epochs
        self.gamma = gamma
        self.GAElambda = GAElambda
        self.epsilon = epsilon
        self.critic_smoother = critic_smoother
        self.eps = eps
        #self.big_T = big_T
        self.entropy_coeff = entropy_coeff
        self.memory = Memory(self.batch_size)
        self.actorNet = actorNet(self.action_dim)
        self.criticNet = criticNet()

        self.actorOptimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.criticOptimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.tfd = tfp.distributions

        self.agent_name = agent_name
        self.model_dir = f"saved_models/{self.agent_name}"
        os.makedirs(self.model_dir, exist_ok=True)

        self.tensor_stack = []


        self.OBJECT_TO_IDX = {
            ' ': 0,
            'X': 1,
            'P': 2,
            'D': 3,
            'O': 4,
            'S': 5,  
        }

        self.HELD_OBJECTS = [
            "dish", "onion","none"
        ]
        self.HELD_OBJECT_TO_IDX = {name: i for i, name in enumerate(self.HELD_OBJECTS)}

        self.ORIENTATION_TO_IDX = {
            'NORTH': 0,
            'SOUTH': 1,
            'EAST': 2,
            'WEST': 3
        }
        self.FIXED_OBJECT_TYPES = ["onion", "dish", "soup", "pot"]  # aggiungi tutti i tipi


    def state_to_tensor(self, state, base_mdp, batch_size=1):
        #instead of pixels we have informations about the objects etc
        target_H, target_W = 15, 15
        grid = np.array(base_mdp.terrain_mtx)
        H, W = grid.shape
        
        if H > target_H or W > target_W:
            raise ValueError(f"Grid ({H}x{W}) is larger than target size ({target_H}x{target_W})")
        
        #10 channels
        num_channels = 10
        tensor = np.zeros((target_H, target_W, num_channels), dtype=np.float32)
        
        #Terrain
        for y in range(H):
            for x in range(W):
                cell = grid[y, x]
                if cell in self.OBJECT_TO_IDX:
                    tensor[y, x, 0] = self.OBJECT_TO_IDX[cell] / len(self.OBJECT_TO_IDX)  # normalized
        
        #Player positions with orientation
        for i, player in enumerate(state.players[:2]):
            x, y = player.position
            orient_val = self.ORIENTATION_TO_IDX.get(player.orientation, 0)
            tensor[y, x, 1 + i] = (orient_val + 1) / 4.0  # normalized to [0.25, 1.0]
        
        #Held objects
        for i, player in enumerate(state.players[:2]):
            x, y = player.position
            if player.has_object():
                obj = player.get_object()
                obj_name = obj.name if obj.name in self.HELD_OBJECT_TO_IDX else "none"
                held_idx = self.HELD_OBJECT_TO_IDX[obj_name]
                tensor[y, x, 3 + i] = (held_idx + 1) / len(self.HELD_OBJECT_TO_IDX)  # normalized
        
        #Loose objects (4 types)
        object_channels = {
            "onion": 5,
            "dish": 6, 
            "soup": 7,
            "pot": 8
        }
        
        # Get all loose objects
        all_objects = []
        if state.objects:
            for value in state.objects.values():
                if isinstance(value, list):
                    all_objects.extend(value)
                else:
                    all_objects.append(value)
        
        for obj in all_objects:
            if obj.name in object_channels:
                x, y = obj.position
                channel = object_channels[obj.name]
                tensor[y, x, channel] = 1.0
        
        for i, player in enumerate(state.players[:2]):
            px, py = player.position

            min_dist = float('inf')
            for y in range(H):
                for x in range(W):
                    if grid[y, x] in ['D', 'S', 'O']:  
                        dist = abs(px - x) + abs(py - y)  
                        min_dist = min(min_dist, dist)
            
            if min_dist != float('inf'):
                # Encode distance at player position
                tensor[py, px, 9] = max(0, 1.0 - min_dist / 10.0)  # closer = higher value
        
            #Stack
            if not hasattr(self, 'tensor_stack'):
                self.tensor_stack = []

            self.tensor_stack.append(tensor)

            # Keep only last 4 tensors (max)
            if len(self.tensor_stack) > 4:
                self.tensor_stack = self.tensor_stack[-4:]

            # If less than 4 frames, pad by repeating the last frame
            current_stack = self.tensor_stack.copy()
            while len(current_stack) < 4:
                current_stack.append(current_stack[-1])  # pad with newest available

            stacked_tensor = np.concatenate(current_stack, axis=-1)  # shape (15, 15, 40)
            self.curr_stacked_tensor = stacked_tensor
            return stacked_tensor
        
    def reset_tensor_stack(self):
        self.tensor_stack = []



    def save_models(self, episode= None, score = None):
        try:

            actor_path = os.path.join(self.model_dir, "actor_model.weights.h5")
            self.actorNet.save_weights(actor_path)
            critic_path = os.path.join(self.model_dir, "ctiric_model.weights.h5")
            self.criticNet.save_weights(critic_path)
            

            info = {
                "episode": episode,
                "score": score,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "gamma": self.gamma,
                "GAElambda": self.GAElambda,
                "epsilon": self.epsilon,
                "entropy_coeff": self.entropy_coeff
            }
            
            info_path = os.path.join(self.model_dir, "training_info.json")
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
                
            print("Modello salvato")
            
        except Exception as e:
            print(f"ERRORE{self.agent_name}: {e}")

    def load_models(self):
        try:
            actor_path = os.path.join(self.model_dir, "actor_model.weights.h5")
            info_path = os.path.join(self.model_dir, "training_info.json")
            critic_path = os.path.join(self.model_dir, "ctiric_model.weights.h5")  
            
            if os.path.exists(actor_path) and os.path.exists(info_path) and os.path.exists(critic_path):
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                # Dummy input per CNN con shape
                #CNN con (15, 15, 40)
                dummy_input = tf.zeros((1, 15, 15, 40), dtype=tf.float32)
                
                # Inizializzazione dei tensori dei pesi per entrambe le reti
                dummy_output = self.actorNet(dummy_input)
                dummy_output = self.criticNet(dummy_input)
                
                # Caricamento dei pesi
                self.actorNet.load_weights(actor_path)
                self.criticNet.load_weights(critic_path)
                
                print(f"Modelli caricati per {self.agent_name}")
                print(f"Ultimo episodio: {info.get('episode', 'N/A')}")
                print(f"Ultimo score: {info.get('score', 'N/A')}")
                
                return info
            else:
                print(f"Nessun modello trovato per {self.agent_name}, iniziando da zero")
                return None
                
        except Exception as e:
            print(f"Errore nel caricamento per {self.agent_name}: {e}")
            return None

    def remember(self,state, action,reward,S_value,old_prob,done):
        self.memory.add_obs(state,action,reward,S_value,old_prob, done)

    def remember_nextS(self,nextS):
        self.memory.add_nextS(nextS)

    def clean(self):
        self.memory.clean_memory()

    
    def generate_action(self, obs): 
        action_probs = []
        obs = tf.expand_dims(obs, axis=0) 

        logits = self.actorNet(obs)
        dist = self.tfd.Categorical(logits = logits)

        for i in range(self.action_dim):
            action_prob = tf.exp(dist.log_prob(i))
            action_probs.append(action_prob.numpy()[0])
            #print(f"Prob azione {i}: {action_prob.numpy()[0]:.4f}")
        #print("DEBUG: ACTION PROBS => ", action_probs)
        action = dist.sample().numpy()[0]
        log_prob = dist.log_prob(action) 
        return action, log_prob
    

    def predict_value(self, obs):
        obs = tf.expand_dims(obs, axis=0) 
        S_value = self.criticNet(obs).numpy()[0,0]
        return S_value
    

    def __computeAdv(self,T, values, rewards, dones, nextS):
        advantages = np.zeros(T,dtype=np.float32)
        advantage = 0.0

        values = np.append(values,nextS)
        for t in reversed(range(T)):
            delta = rewards[t] + (1-dones[t]) * self.gamma * values[t + 1] - values[t]
            advantage = delta + (1-dones[t]) * self.GAElambda * self.gamma * advantage
            advantages[t] = advantage

        return tf.convert_to_tensor(advantages, dtype=tf.float32)
    

    def use_computeAdv(self, T, values, rewards, dones,nextS):
        return self.__computeAdv(T,values,rewards, dones, nextS)


    def train(self):
        states, actions,rewards, S_values,old_probs,dones,nextS= self.memory.return_np_arrays()
        
        GAE_adv = self.use_computeAdv(len(rewards), S_values, rewards, dones, nextS) #tensore con GAE dentro
        #GAE_adv = (GAE_adv - tf.reduce_mean(GAE_adv))  / (tf.math.reduce_std(GAE_adv) + 1e-8) #normalizziamo o potrebbe diventare troppo grande
        #- tf.reduce_mean(GAE_adv))
        for i in range(self.epochs):

            batches = self.memory.return_batch()

            min_clip = 1 - self.epsilon
            max_clip = 1 + self.epsilon

            for batch in batches:
                b_states = states[batch]
                b_old_probs = old_probs[batch]
                b_actions = actions[batch]
                #b_rewards = rewards[batch]
                b_S_values = S_values[batch]

                t_b_states = tf.convert_to_tensor(b_states, dtype = tf.float32)
                t_b_actions = tf.convert_to_tensor(b_actions, dtype = tf.int32)
                t_b_old_probs = tf.squeeze(tf.convert_to_tensor(b_old_probs, dtype=tf.float32))
                t_b_S_values = tf.convert_to_tensor(b_S_values, dtype = tf.float32)

                with tf.GradientTape(persistent=True) as tape:
                    new_logits = self.actorNet(t_b_states)
                    new_dist = self.tfd.Categorical(logits = new_logits)
                    entropy = tf.reduce_mean(new_dist.entropy())
                    new_probs = new_dist.log_prob(t_b_actions)

                    critic_value = tf.squeeze(self.criticNet(t_b_states), axis=-1)

                    prob_ratio = tf.exp(new_probs - t_b_old_probs) 
                    #log/log = exp (log - log)
                    adv_prob_ratio = prob_ratio * tf.gather(GAE_adv, batch)

                    clipped_ratio = tf.clip_by_value(prob_ratio, min_clip, max_clip)
                    adv_clipped_ratio = clipped_ratio * tf.gather(GAE_adv, batch)
                

                    #computing the 2 losses
                    policy_loss = -tf.reduce_mean(tf.minimum(adv_prob_ratio, adv_clipped_ratio)) 
                    actor_loss = policy_loss - (self.entropy_coeff * entropy)
                    # Critic loss
                    
                    
                    returns = tf.gather(GAE_adv, batch) + t_b_S_values 
                    returns_clipped = tf.gather(GAE_adv, batch)  + tf.clip_by_value(critic_value - t_b_S_values, -self.eps, self.eps)
                    critic_loss = self.critic_smoother * tf.reduce_mean(tf.maximum(tf.square(returns - critic_value),tf.square(returns_clipped - critic_value)))

                #print(f"[Ratio] mean={tf.reduce_mean(prob_ratio):.4f}, max={tf.reduce_max(prob_ratio):.4f}, min={tf.reduce_min(prob_ratio):.4f}")

                actor_grads = tape.gradient(actor_loss, self.actorNet.trainable_variables)
                critic_grads = tape.gradient(critic_loss, self.criticNet.trainable_variables)

                self.actorOptimizer.apply_gradients(zip(actor_grads, self.actorNet.trainable_variables))
                self.criticOptimizer.apply_gradients(zip(critic_grads, self.criticNet.trainable_variables))

                del tape
            if i == 5:
                print(f"Policy loss: {actor_loss:.4f}")
                print(f"Critic loss: {critic_loss:.4f}")
                print(f"entropy: {entropy:.4f}")
            
        self.clean()    

            






            
            


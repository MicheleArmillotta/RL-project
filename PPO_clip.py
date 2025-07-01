import os
import numpy as np
import tensorflow as tf
from keras import layers
import tensorflow_probability as tfp
import json


#PPO implementation -> we need
# - memory class
# - 2 distinct net class -> actor net and critic net

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

    def return_batch(self, shuffle = True):  #we need a shuffled minibatch
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
        return np.array(self.states),np.array(self.actions),np.array(self.rewards).flatten(),\
        np.array(self.S_values).flatten(),np.array(self.old_probs),np.array(self.dones).flatten(), self.nextS
    

class actorNet(tf.keras.Model):
    def __init__(self, input_dim, action_dim):
        super(actorNet, self).__init__()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.logits = layers.Dense(action_dim) 
        self.dropout = layers.Dropout(0.1) #could prevent overfitting
        #self.softmax = tf.nn.softmax(logits,axis=None, name=None)

    def call(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return self.logits(x) #valori non normalizzati input per softmax
        #return tf.nn.softmax(x, axis=-1)  #attenzione(?)
        #return self.logits(x)
    




class criticNet(tf.keras.Model):
    def __init__(self, input_dim):
        super(criticNet, self).__init__()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.logits = layers.Dense(1) 
        self.dropout = layers.Dropout(0.1)

    def call(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.logits(x)
    


class Agent:
    def __init__(self, batch_size, input_dim, action_dim, epochs, gamma, GAElambda, epsilon, learning_rate,entropy_coeff, critic_smoother, agent_name = "Agent"):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.action_dim = action_dim
        #molte piu cose da aggiungere qui
        self.epochs = epochs
        self.gamma = gamma
        self.GAElambda = GAElambda
        self.epsilon = epsilon
        self.critic_smoother = critic_smoother
        #self.big_T = big_T
        self.entropy_coeff = entropy_coeff
        self.memory = Memory(self.batch_size)
        self.actorNet = actorNet(self.input_dim, self.action_dim)
        self.criticNet = criticNet(self.input_dim)

        self.actorOptimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.criticOptimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.tfd = tfp.distributions

        self.agent_name = agent_name
        self.model_dir = f"saved_models/{self.agent_name}"
        os.makedirs(self.model_dir, exist_ok=True)

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
                
                dummy_input = tf.zeros((1, int(self.input_dim[0])), dtype=tf.float32)

                #per inizializzazione dei tensori dei pesi altrimenti non si caricano con load_weights
                dummy_output = self.actorNet(dummy_input)
                dummy_output = self.criticNet(dummy_input)
                

                self.actorNet.load_weights(actor_path)
                self.criticNet.load_weights(critic_path)
                
                print(f"Modell caricat per {self.agent_name}")
                print(f" Ultimo episodio: {info.get('episode', 'N/A')}")
                print(f"Ultimo score: {info.get('score', 'N/A')}")
                
                return info
            else:
                print(f"Nessun modello  per {self.agent_name}, iniziando da zero")
                return None
                
        except Exception as e:
            print(f" Errore {self.agent_name}: {e}")
            return None


        #nel loop bisogna
        # 1. fare le osservazioni
        # 2. aggiornare la memooria
        # 3. calcolare l'advantage (con loss)
        #       Ogni quanto aggiorna rete V(t)?
        # 4. calcolare il rapporto con vecchia rete
        # 5. cambiare vecchia rete con nuova aggiornata


        #OK IO IN QUESTA CLASSE VOGLIO SOLO TRAIN COME SE AVESSE GIA FATTO DELLE OSSERVAZIONI (da fillare nel quando si usa)
        #DEVO DARE UN MODO PER SCEGLIERE LE AZIONI E OTTENERE STATI ECC ECC
        #DEVO DARE UN MODO PER RICORDARE E NON UTILIZZARE DIRETTAMENTE MEMORY

    def remember(self,state, action,reward,S_value,old_prob,done):
        self.memory.add_obs(state,action,reward,S_value,old_prob, done)

    def remember_nextS(self,nextS):
        self.memory.add_nextS(nextS)

    def clean(self):
        self.memory.clean_memory()

    
    def generate_action(self, obs): #obs è un array, da qui voglio ottenere -> action, old_prob e S_value
        #obs_array = obs[0]  # prende solo l'array, scarta il dizionario vuoto

        t_obs = tf.convert_to_tensor([obs], dtype=tf.float32)

        #scelgo le azioni
        #æctor

        logits = self.actorNet(t_obs)
        dist = self.tfd.Categorical(logits = logits)#azione scelta dalla dist
        action = dist.sample().numpy()[0] # -> potrei dover mettere [0,0] / scelta causaleponderata
        log_prob = dist.log_prob(action) #log_probabilità per PPO

        return action, log_prob
    

    def predict_value(self, obs):
        t_obs = tf.convert_to_tensor([obs], dtype=tf.float32)
        S_value = self.criticNet(t_obs).numpy()[0,0]
        return S_value
    

    def __computeAdv(self,T, values, rewards, dones, nextS):
        advantages = np.zeros(T,dtype=np.float32)
        advantage = 0.0

        values = np.append(values,nextS)
        for t in reversed(range(T)): #uso reversed cosi non devo fare esponenziale
            delta = rewards[t] + (1-dones[t]) * self.gamma * values[t + 1] - values[t]
            advantage = delta + (1-dones[t]) * self.GAElambda * self.gamma * advantage
            advantages[t] = advantage

            #ricorsività -> calcolo delta caso base ricorsività riga dopo -> mettere primo all'iltimo!!

        return tf.convert_to_tensor(advantages, dtype=tf.float32)
    

    def use_computeAdv(self, T, values, rewards, dones,nextS):
        return self.__computeAdv(T,values,rewards, dones, nextS)


    def train(self):
        states,actions,rewards, S_values,old_probs,dones,nextS= self.memory.return_np_arrays()
        GAE_adv = self.use_computeAdv(len(rewards), S_values, rewards, dones, nextS) #tensore con GAE dentro
        #GAE_adv = (GAE_adv - tf.reduce_mean(GAE_adv)) / (tf.math.reduce_std(GAE_adv) + 1e-8) #normalizziamo o potrebbe diventare troppo grande

        for i in range(self.epochs):
            #devo calcolare advantage, actor loss, critic loss e rapporto
            #iniziamo con l'advantage
            batches = self.memory.return_batch()
            #states,actions,rewards, S_values,old_probs,dones,nextS= self.memory.return_np_arrays()

            min_clip = 1 - self.epsilon
            max_clip = 1 + self.epsilon
            
            #forse mi serve tensore per l'advantage
            #GAE_adv = self.use_computeAdv(len(rewards), S_values, rewards, dones, nextS) #tensore con GAE dentro
            #print(f"[Advantage] mean: {tf.reduce_mean(GAE_adv):.4f}, std: {tf.math.reduce_std(GAE_adv):.4f}, max: {tf.reduce_max(GAE_adv):.4f}, min: {tf.reduce_min(GAE_adv):.4f}")



            for batch in batches:
                b_states = states[batch]
                b_old_probs = old_probs[batch]
                b_actions = actions[batch]
                #b_rewards = rewards[batch]
                b_S_values = S_values[batch] #stare ATTENTI -> in fase di training dall'altra parte salvare uno stato in piu


                #ottengo tensori

                t_b_states = tf.convert_to_tensor(b_states, dtype = tf.float32)
                t_b_actions = tf.convert_to_tensor(b_actions, dtype = tf.int32)
                t_b_old_probs = tf.convert_to_tensor(b_old_probs, dtype = tf.float32)
                t_b_S_values = tf.convert_to_tensor(b_S_values, dtype = tf.float32)

                with tf.GradientTape(persistent=True) as tape:
                    new_logits = self.actorNet(t_b_states)
                    new_dist = self.tfd.Categorical(logits = new_logits)#azione scelta dalla dist
                    entropy = tf.reduce_mean(new_dist.entropy())
                    new_probs = new_dist.log_prob(t_b_actions)

                    critic_value = self.criticNet(t_b_states)

                    #ora posso calcolare la differenza e tutto il cuore di PPO

                    prob_ratio = tf.exp(new_probs - t_b_old_probs) #IMPORTANTE, PER ORA LASCIAMO COSI, POTREI VOLER SALVARE DIRETTAMENTE LA VECCHIA POLICy (I PESI DELLA VECCHIA RETE) 
                    #log/log = exp (log - log)
                    adv_prob_ratio = prob_ratio * tf.gather(GAE_adv, batch)

                    clipped_ratio = tf.clip_by_value(prob_ratio, min_clip, max_clip)
                    adv_clipped_ratio = clipped_ratio * tf.gather(GAE_adv, batch)

                    #computing the 2 losses
                    # Actor loss (nota il segno meno per gradient descent)
                    policy_loss = -tf.reduce_mean(tf.minimum(adv_prob_ratio, adv_clipped_ratio)) #per gradient descent
                    actor_loss = policy_loss - (self.entropy_coeff * entropy)
                    # Critic loss
                    returns = tf.gather(GAE_adv, batch) + t_b_S_values # critic_value è V(s) con la nuova rete
                    critic_loss = self.critic_smoother * tf.reduce_mean(tf.square(returns - critic_value))
                #print(f"[Ratio] mean={tf.reduce_mean(prob_ratio):.4f}, max={tf.reduce_max(prob_ratio):.4f}, min={tf.reduce_min(prob_ratio):.4f}")

                actor_grads = tape.gradient(actor_loss, self.actorNet.trainable_variables)
                critic_grads = tape.gradient(critic_loss, self.criticNet.trainable_variables)

                self.actorOptimizer.apply_gradients(zip(actor_grads, self.actorNet.trainable_variables))
                self.criticOptimizer.apply_gradients(zip(critic_grads, self.criticNet.trainable_variables))

                del tape
            if i == 9:
                print(f"Policy loss: {actor_loss:.4f}")
                print(f"Critic loss: {critic_loss:.4f}")
                print(f"entropy: {entropy:.4f}")
            
        self.clean()    

            






            
            


import os
import numpy as np
import tensorflow as tf
from keras import layers
import tensorflow_probability as tfp


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

    def return_batch(self, data_size,shuffle = True):  #we need a shuffled minibatch

        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)

        batches = []
        for start in range(0, data_size, self.batch_size):
            end = start + self.batch_size
            batches.append(indices[start:end])

        return batches
    

    def return_np_arrays(self):
        return np.array(self.states),np.array(self.actions),\
        np.array(self.old_probs)
    

class actorNet(tf.keras.Model):
    def __init__(self, input_dim, action_dim):
        super(actorNet, self).__init__()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.logits = layers.Dense(action_dim) 
        self.dropout = layers.Dropout(0.1)
        #self.softmax = tf.nn.softmax(logits,axis=None, name=None)

    def call(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return self.logits(x)
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
    
class critic_Train:
    def __init__(self,input_dim,criticNet,epochs,learning_rate,gamma,GAElambda,critic_smoother,eps):
        self.criticNet = criticNet
        self.epochs = epochs
        
        #self.T = T
        self.input_dim = input_dim
        self.gamma = gamma
        self.GAElambda = GAElambda
        self.critic_smoother = critic_smoother
        self.eps = eps
        self.criticOptimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.model_dir = f"saved_models/critic"
        os.makedirs(self.model_dir, exist_ok=True)

    def save_models(self, episode= None, score = None):
        try:
           
            actor_path = os.path.join(self.model_dir, "critic_model.weights.h5")
            self.criticNet.save_weights(actor_path)
            
            # Salva informazioni aggiuntive
            info = {
                "episode": episode,
                "score": score,
                "epochs": self.epochs,
                "gamma": self.gamma,
                "GAElambda": self.GAElambda,
            }
            
            info_path = os.path.join(self.model_dir, "training_info.json")
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
                
            print("Modello critic salvato")
            
        except Exception as e:
            print(f"ERRORE salvataggiocritic: {e}")

    def load_models(self):
        try:
            critic_path = os.path.join(self.model_dir, "critic_model.weights.h5")
            info_path = os.path.join(self.model_dir, "training_info.json")
            
            if os.path.exists(critic_path) and os.path.exists(info_path):
                # Carica le informazioni
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                # Inizializza le reti con un forward pass prima di caricare i pesi
                #print(f"self.input_dim: {self.input_dim}, type: {type(self.input_dim)}")
                dummy_input = tf.zeros((1, int(self.input_dim[0])), dtype=tf.float32)

# Esegui una forward pass per inizializzare i pesi
                dummy_output = self.criticNet(dummy_input)
                
                print(f"Modell caricat per critic")
                print(f" Ultimo episodio: {info.get('episode', 'N/A')}")
                print(f"Ultimo score: {info.get('score', 'N/A')}")
                
                return info
            else:
                print(f"Nessun modello  per critic, iniziando da zero")
                return None
                
        except Exception as e:
            print(f" Errore critic load: {e}")
            return None

    def computeAdv(self,T, values, rewards, dones, nextS):
        advantages = np.zeros(T,dtype=np.float32)
        advantage = 0.0

        values = np.append(values,nextS)
        for t in reversed(range(T)): #uso reversed cosi non devo fare esponenziale
            delta = rewards[t] + (1-dones[t]) * self.gamma * values[t + 1] - values[t]
            advantage = delta + (1-dones[t]) * self.GAElambda * self.gamma * advantage
            advantages[t] = advantage

        return tf.convert_to_tensor(advantages, dtype=tf.float32)
    
    def train_critic(self, memory0, memory1):
        
        adv0 = self.computeAdv(len(memory0.rewards),memory0.S_values,memory0.rewards,memory0.dones,memory0.nextS)
        adv1 = self.computeAdv(len(memory1.rewards),memory1.S_values,memory1.rewards,memory1.dones,memory1.nextS)

        shared_states = np.concatenate([memory0.states,memory1.states])
        shared_values = np.concatenate([memory0.S_values,memory1.S_values])
        shared_advantages = tf.concat([adv0,adv1], axis=0)
        batches = memory0.return_batch(len(shared_states))

        for i in range(self.epochs):
            for batch in batches:
                b_S_values = shared_values[batch]
                t_b_S_values = tf.convert_to_tensor(b_S_values, dtype = tf.float32)
                b_S_states = shared_states[batch]
                t_b_states = tf.convert_to_tensor(b_S_states, dtype = tf.float32)
                with tf.GradientTape(persistent=True) as tape:
                    critic_value = self.criticNet(t_b_states)
                    returns = tf.gather(shared_advantages, batch) + t_b_S_values # critic_value è V(s) con la nuova rete
                    returns_clipped = tf.gather(shared_advantages, batch)  + tf.clip_by_value(critic_value - tf.gather(t_b_S_values, batch), -self.eps, self.eps)
                    critic_loss = self.critic_smoother * tf.reduce_mean(tf.maximum(tf.square(returns - critic_value),tf.square(returns_clipped - critic_value)))
                critic_grads = tape.gradient(critic_loss, self.criticNet.trainable_variables)
                self.criticOptimizer.apply_gradients(zip(critic_grads, self.criticNet.trainable_variables))
                del tape
            if i == 9:
                print(f"Critic loss: {critic_loss:.4f}")

        return adv0,adv1

    


class Agent:
    def __init__(self, criticNet,batch_size, input_dim, action_dim, epochs, gamma, GAElambda, epsilon, learning_rate,entropy_coeff, agent_name="agent"):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.agent_name = agent_name  # Nome per identificare i modelli salvati
        #molte piu cose da aggiungere qui
        self.epochs = epochs
        self.gamma = gamma
        self.GAElambda = GAElambda
        self.epsilon = epsilon
        #self.big_T = big_T
        self.entropy_coeff = entropy_coeff
        self.memory = Memory(self.batch_size)
        self.actorNet = actorNet(self.input_dim, self.action_dim)
        self.criticNet = criticNet

        self.actorOptimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.criticOptimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.tfd = tfp.distributions

        # Directory per salvare i modelli
        self.model_dir = f"saved_models/{self.agent_name}"
        os.makedirs(self.model_dir, exist_ok=True)

        self.actorOptimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.criticOptimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.tfd = tfp.distributions

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

    def save_models(self, episode= None, score = None):
        try:
            # Salva actor network
            actor_path = os.path.join(self.model_dir, "actor_model.weights.h5")
            self.actorNet.save_weights(actor_path)
            
            # Salva informazioni aggiuntive
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
            
            if os.path.exists(actor_path) and os.path.exists(info_path):
                # Carica le informazioni
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                dummy_input = tf.zeros((1, int(self.input_dim[0])), dtype=tf.float32)


                dummy_output = self.actorNet(dummy_input)
                
                # Carica i pesi
                self.actorNet.load_weights(actor_path)
                
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
        action = dist.sample().numpy()[0] # -> potrei dover mettere [0,0]
        log_prob = dist.log_prob(action)

        return action, log_prob
    

    
    def predict_value(self, obs):
        t_obs = tf.convert_to_tensor([obs], dtype=tf.float32)
        S_value = self.criticNet(t_obs).numpy()[0,0]
        return S_value

    def train(self,GAE_adv):
        states,actions,old_probs= self.memory.return_np_arrays()
        #GAE_adv = self.use_computeAdv(len(rewards), S_values, rewards, dones, nextS) #tensore con GAE dentro
        #GAE_adv = (GAE_adv - tf.reduce_mean(GAE_adv)) / (tf.math.reduce_std(GAE_adv) + 1e-4) #normalizziamo o potrebbe diventare troppo grande

        print(f"[Advantage Stats] mean: {tf.reduce_mean(GAE_adv):.4f}, "
          f"std: {tf.math.reduce_std(GAE_adv):.4f}, "
          f"max: {tf.reduce_max(GAE_adv):.4f}, "
          f"min: {tf.reduce_min(GAE_adv):.4f}")


        total_policy_loss = 0
        total_entropy = 0
        total_clip_fraction = 0

        for i in range(self.epochs):
            #devo calcolare advantage, actor loss, critic loss e rapporto
            #iniziamo con l'advantage
            batches = self.memory.return_batch(len(states))
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
                #b_S_values = S_values[batch] #stare ATTENTI -> in fase di training dall'altra parte salvare uno stato in piu


                #ottengo tensori

                t_b_states = tf.convert_to_tensor(b_states, dtype = tf.float32)
                t_b_actions = tf.convert_to_tensor(b_actions, dtype = tf.int32)
                t_b_old_probs = tf.convert_to_tensor(b_old_probs, dtype = tf.float32)
                #t_b_S_values = tf.convert_to_tensor(b_S_values, dtype = tf.float32)

                with tf.GradientTape(persistent=True) as tape:
                    new_logits = self.actorNet(t_b_states)
                    new_dist = self.tfd.Categorical(logits = new_logits)#azione scelta dalla dist
                    entropy = tf.reduce_mean(new_dist.entropy())
                    new_probs = new_dist.log_prob(t_b_actions)

                    
                    #ora posso calcolare la differenza e tutto il cuore di PPO

                    prob_ratio = tf.exp(new_probs - t_b_old_probs) #IMPORTANTE, PER ORA LASCIAMO COSI, POTREI VOLER SALVARE DIRETTAMENTE LA VECCHIA POLICy (I PESI DELLA VECCHIA RETE)
                    adv_prob_ratio = prob_ratio * tf.gather(GAE_adv, batch)

                    clipped_ratio = tf.clip_by_value(prob_ratio, min_clip, max_clip)
                    adv_clipped_ratio = clipped_ratio * tf.gather(GAE_adv, batch)
                    
                    clip_fraction = tf.reduce_mean(
                    tf.cast(tf.abs(prob_ratio - 1.0) > self.epsilon, tf.float32)
                    )
                    total_clip_fraction += clip_fraction
                    #computing the 2 losses
                    # Actor loss (nota il segno meno per gradient descent)
                    policy_loss = -tf.reduce_mean(tf.minimum(adv_prob_ratio, adv_clipped_ratio))
                    actor_loss = policy_loss - (self.entropy_coeff * entropy)
                    
                #print(f"[Ratio] mean={tf.reduce_mean(prob_ratio):.4f}, max={tf.reduce_max(prob_ratio):.4f}, min={tf.reduce_min(prob_ratio):.4f}")
                total_policy_loss += policy_loss
                total_entropy += entropy                 
                actor_grads = tape.gradient(actor_loss, self.actorNet.trainable_variables)


                self.actorOptimizer.apply_gradients(zip(actor_grads, self.actorNet.trainable_variables))

                del tape
            if i == 9:
                print(f"Policy loss: {actor_loss:.4f}")
                #print(f"Critic loss: {critic_loss:.4f}")
                print(f"entropy: {entropy:.4f}")
                avg_policy_loss = total_policy_loss / (len(batches) * self.epochs)
                avg_entropy = total_entropy / (len(batches) * self.epochs)
                avg_clip_fraction = total_clip_fraction / (len(batches) * self.epochs)
            
                print(f"Policy loss: {avg_policy_loss:.4f}, "
                  f"Entropy: {avg_entropy:.4f}, "
                  )
            
        self.clean()    

            






            
            


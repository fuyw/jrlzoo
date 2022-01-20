from models import DynamicsModel

env = 'hopper-medium-v2'
seed = 1
elite_num = 5
ensemble_num = 7

# Load pre-trained model
model = DynamicsModel(env, seed, ensemble_num, elite_num)
model.load('ensemble_models/hopper-medium-v2/s1')

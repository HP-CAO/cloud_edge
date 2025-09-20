import pickle
from matplotlib import pyplot as plt

path = "logs/remote_training/training/training_trajectories.p"

data = pickle.load(open(path, "rb"))
print(data.keys())
# print(data['rewards'])
print(data['critic_losses'])

plt.plot(data['rewards'])
plt.xlabel('steps')
plt.ylabel('Reward')
plt.title('Training Rewards Over Episodes')
plt.show()

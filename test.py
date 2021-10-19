from realips.agent.base import BaseAgent, BaseAgentParams

agent = BaseAgent(BaseAgentParams(), 10)
actor_a = agent.build_actor("a")
actor_b = agent.build_actor("b")

actor_b_weights = actor_b.get_weights()

weights_lst = [actor_b.layers[i + 3].get_weights() for i in range(len(actor_b.layers) - 3)]


for i, weights in enumerate(actor_b_weights):
    actor_a.weights[i].assign(weights)

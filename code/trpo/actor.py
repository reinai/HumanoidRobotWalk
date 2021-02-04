import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from critic import Critic


HIDDEN_LAYER_SIZE = 64
INPUT_SIZE = 44 # state size
OUTPUT_LAYER_SIZE = 17 # probability for each action (num of actions)

class ActorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.fc3 = nn.Linear(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class Actor():
    def __init__(self, delta):
        self.model = ActorModel()
        self.critic = Critic()
        self.delta = delta

    def get_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0)
        dist = Categorical(torch.distributions.utils.clamp_probs(self.model.forward(state)))
        return dist.sample().item()

    def update_actor(self, episodes):
        states = torch.cat([r.states for r in episodes], dim=0)
        actions = torch.cat([r.actions for r in episodes], dim=0).flatten()

        def estimate_advantages(states, last_state, rewards):
            values = self.critic.model(states)
            last_value = self.critic.model(last_state.unsqueeze(0))
            next_values = torch.zeros_like(rewards)
            for i in reversed(range(rewards.shape[0])):
                last_value = next_values[i] = rewards[i] + last_value
            advantages = next_values - values
            return advantages
        advantages = [estimate_advantages(states, next_states[-1], rewards) for states, _, rewards, next_states in episodes]
        advantages = torch.cat(advantages, dim=0).flatten()

        # Normalize advantages to reduce skewness and improve convergence
        advantages = (advantages - advantages.mean()) / advantages.std()

        self.critic.update_critic(advantages)

        distribution = self.model.forward(states)
        distribution = torch.distributions.utils.clamp_probs(distribution)
        probabilities = distribution[range(distribution.shape[0]), actions]

        def surrogate_loss(new_probs, old_probs, advantages):
            return (new_probs / old_probs * advantages).mean()
        L = surrogate_loss(probabilities, probabilities.detach(), advantages)
        print('L',L)

        def kl_divergence(p, q):
            p = p.detach()
            return (p * (p.log() - q.log())).sum(-1).mean()
        KL = kl_divergence(distribution, distribution)
        print('KL', KL)

        def flat_grad(y, x, retain_graph=False, create_graph=False):
            if create_graph:
                retain_graph = True
            g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
            g = torch.cat([t.view(-1) for t in g])
            return g
        parameters = list(self.model.parameters())
        g = flat_grad(L, parameters, retain_graph=True)
        d_kl = flat_grad(KL, parameters, create_graph=True)

        def HVP(v):
            return flat_grad(d_kl @ v, parameters, retain_graph=True)
        def conjugate_gradient(A, b, delta=0., max_iterations=10):
            x = torch.zeros_like(b)
            r = b.clone()
            p = b.clone()
            i = 0
            while i < max_iterations:
                AVP = A(p)
                dot_old = r @ r
                alpha = dot_old / (p @ AVP)
                x_new = x + alpha * p
                if (x - x_new).norm() <= delta:
                    return x_new
                i += 1
                r = r - alpha * AVP
                beta = (r @ r) / dot_old
                p = r + beta * p
                x = x_new
            return x
        search_dir = conjugate_gradient(HVP, g)
        max_length = torch.sqrt(2 * self.delta / (search_dir @ HVP(search_dir)))
        max_step = max_length * search_dir

        def criterion(step):
            self.upgrade_parameters(step)
            with torch.no_grad():
                distribution_new = self.model.forward(states)
                distribution_new = torch.distributions.utils.clamp_probs(distribution_new)
                probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]
                L_new = surrogate_loss(probabilities_new, probabilities, advantages)
                KL_new = kl_divergence(distribution, distribution_new)
            L_improvement = L_new - L
            if L_improvement > 0 and KL_new <= self.delta:
                return True
            self.upgrade_parameters(-step)
            return False
        i = 0
        while not criterion((0.9 ** i) * max_step) and i < 10:
            i += 1

    def upgrade_parameters(self, grads):
        n = 0
        for p in self.model.parameters():
            numel = p.numel()
            g = grads[n:n + numel].view(p.shape)
            p.data += g
            n += numel

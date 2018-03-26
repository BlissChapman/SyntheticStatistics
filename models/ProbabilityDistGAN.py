import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad


class Generator(nn.Module):
    """
    """

    def __init__(self, input_width, output_width, dimensionality, cudaEnabled):
        super(Generator, self).__init__()

        self.fc_1 = nn.Linear(input_width, dimensionality)
        self.fc_2 = nn.Linear(dimensionality, dimensionality)
        self.fc_3 = nn.Linear(dimensionality, dimensionality)
        self.fc_4 = nn.Linear(dimensionality, dimensionality)
        self.fc_5 = nn.Linear(dimensionality, output_width)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.cudaEnabled = cudaEnabled

        if self.cudaEnabled:
            self.cuda()

    def forward(self, noise):
        out = self.fc_1(noise)
        out = F.leaky_relu(out, inplace=True)

        out = self.fc_2(out)
        out = F.leaky_relu(out, inplace=True)

        out = self.fc_3(out)
        out = F.leaky_relu(out, inplace=True)

        out = self.fc_4(out)
        out = F.leaky_relu(out, inplace=True)

        out = self.fc_5(out)
        return out

    def train(self, critic_outputs):
        self.zero_grad()
        g_loss = -torch.mean(critic_outputs)
        g_loss.backward()
        self.optimizer.step()
        return g_loss


class Critic(nn.Module):
    """
    """

    def __init__(self, input_width, dimensionality, cudaEnabled):
        super(Critic, self).__init__()

        self.fc_1 = nn.Linear(input_width, dimensionality)
        self.fc_2 = nn.Linear(dimensionality, dimensionality)
        self.fc_3 = nn.Linear(dimensionality, dimensionality)
        self.fc_4 = nn.Linear(dimensionality, dimensionality)
        self.fc_5 = nn.Linear(dimensionality, 1)

        self.cudaEnabled = cudaEnabled
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=(0.5, 0.9))

        if self.cudaEnabled:
            self.cuda()

    def forward(self, data):
        out = self.fc_1(data)
        out = F.leaky_relu(out, inplace=True)

        out = self.fc_2(out)
        out = F.leaky_relu(out, inplace=True)

        out = self.fc_3(out)
        out = F.leaky_relu(out, inplace=True)

        out = self.fc_4(out)
        out = F.leaky_relu(out, inplace=True)

        out = self.fc_5(out)
        return out

    def train(self, real_data, synthetic_data, LAMBDA):
        # Housekeeping
        self.zero_grad()

        # Compute gradient penalty:
        random_samples = torch.rand(real_data.size())
        interpolated_random_samples = random_samples * real_data.data.cpu() + ((1 - random_samples) * synthetic_data.data.cpu())
        interpolated_random_samples = Variable(interpolated_random_samples, requires_grad=True)
        if self.cudaEnabled:
            interpolated_random_samples = interpolated_random_samples.cuda()

        critic_random_sample_output = self(interpolated_random_samples)
        grad_outputs = torch.ones(critic_random_sample_output.size())
        if self.cudaEnabled:
            grad_outputs = grad_outputs.cuda()

        gradients = grad(outputs=critic_random_sample_output,
                         inputs=interpolated_random_samples,
                         grad_outputs=grad_outputs,
                         create_graph=True,
                         retain_graph=True,
                         only_inputs=True)[0]
        if self.cudaEnabled:
            gradients = gradients.cuda()

        gradient_penalty = LAMBDA * ((gradients.norm(2) - 1) ** 2).mean()
        if self.cudaEnabled:
            gradient_penalty = gradient_penalty.cuda()

        # Critic output:
        critic_real_output = self(real_data)
        critic_synthetic_output = self(synthetic_data)

        # Compute loss
        critic_loss = -(torch.mean(critic_real_output) - torch.mean(critic_synthetic_output)) + gradient_penalty
        critic_loss.backward()

        # Optimize critic's parameters
        self.optimizer.step()

        return critic_loss

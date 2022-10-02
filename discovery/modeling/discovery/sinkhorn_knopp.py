import torch
import torch.distributed as dist


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, logits):
        Q = torch.exp(logits / self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]

        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.num_iters):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B
        return Q.t()


class SinkhornKnoppLognormalPrior(torch.nn.Module):
    def __init__(self, temp, gauss_sd, lamb):
        super().__init__()
        self.temp = temp
        self.gauss_sd = gauss_sd
        self.lamb = lamb
        self.dist = None

    @torch.no_grad()
    def forward(self, logits):
        PS = torch.nn.functional.softmax(logits / self.temp, dim=1, dtype=torch.float64)

        N = PS.size(0)
        K = PS.size(1)
        _K_dist = torch.ones((K, 1), dtype=torch.float64).cuda()  # / K
        marginals_argsort = torch.argsort(PS.sum(0))
        if self.dist is None:
            _K_dist = torch.distributions.log_normal.LogNormal(torch.tensor([1.0]), torch.tensor([self.gauss_sd])).sample(sample_shape=(K, 1)).reshape(-1, 1).cuda() * N / K
            _K_dist = torch.clamp(_K_dist, min=1)
            self.dist = _K_dist
        else:
            _K_dist = self.dist
        _K_dist[marginals_argsort] = torch.sort(_K_dist)[0]

        beta = torch.ones((N, 1), dtype=torch.float64).cuda() / N
        PS.pow_(0.5 * self.lamb)
        r = 1. / _K_dist
        r /= r.sum()

        c = 1. / N
        err = 1e6
        _counter = 0

        ones = torch.ones(N, dtype=torch.float64).cuda()
        while (err > 1e-1) and (_counter < 2000):
            alpha = r / torch.matmul(beta.t(), PS).t()
            beta_new = c / torch.matmul(PS, alpha)
            if _counter % 10 == 0:
                err = torch.sum(torch.abs((beta.squeeze() / beta_new.squeeze()) - ones)).cpu().item()
            beta = beta_new
            _counter += 1

        # inplace calculations
        torch.mul(PS, beta, out=PS)
        torch.mul(alpha.t(), PS, out=PS)

        PS = PS / torch.sum(PS, dim=1, keepdim=True)
        PS = PS.to(torch.half)

        return PS

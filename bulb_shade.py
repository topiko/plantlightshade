import argparse
import torch
import math
import matplotlib.pyplot as plt

TENSOR_PI = torch.tensor(math.pi)
ALPHA = 0.1
NUM_BINS = 100
SOURCE_H = 0.3
REFLECTION_BINS = torch.linspace(0, math.pi, NUM_BINS)


def cast(points_and_angles: torch.tensor, lens: torch.tensor) -> torch.tensor:
    points = points_and_angles[:, :2]
    angles = points_and_angles[:, 2]

    dxs = lens * torch.cos(angles)
    dys = lens * torch.sin(angles)

    return points + torch.vstack([dxs, dys]).T


def get_target(num_rays: int) -> torch.tensor:
    return torch.linspace(TENSOR_PI / 2, TENSOR_PI / 3, num_rays, dtype=torch.float)


class Mirror:
    def __init__(self, num_rays: int, source_h: float, width: float = 1.0):
        num_pieces = num_rays
        self.num_pieces = num_pieces
        self._xs = torch.linspace(width / 1e12, width, num_pieces, requires_grad=False)
        self._ys = torch.zeros(num_pieces, requires_grad=True)
        self.source_h = source_h

    @property
    def surface(self) -> torch.tensor:
        return torch.vstack([self._xs, self._ys]).T

    def plot(self, source: bool = True):
        surf = self.surface.detach().numpy()

        plt.plot(surf[:, 0], surf[:, 1], marker="o")

        if source:
            incoming_rays, outgoing_rays = self.reflect()
            incoming_lens = (
                incoming_rays[:, 0] ** 2 + (self.source_h - incoming_rays[:, 1]) ** 2
            ) ** 0.5

            incoming_endpoints = cast(incoming_rays, incoming_lens).detach().numpy()
            outgoing_endpoints = cast(outgoing_rays, incoming_lens).detach().numpy()
            incoming_rays = incoming_rays.detach().numpy()
            outgoing_rays = outgoing_rays.detach().numpy()

            for i in range(incoming_rays.shape[0]):
                plt.plot(
                    [incoming_rays[i, 0], incoming_endpoints[i, 0]],
                    [incoming_rays[i, 1], incoming_endpoints[i, 1]],
                    color="blue",
                    alpha=ALPHA,
                )

                plt.plot(
                    [outgoing_rays[i, 0], outgoing_endpoints[i, 0]],
                    [outgoing_rays[i, 1], outgoing_endpoints[i, 1]],
                    color="red",
                    alpha=ALPHA,
                )

            ax = plt.gca().inset_axes([0.0, 0.8, 0.2, 0.2])
            ax.hist(self.reflection_angle_distr().detach().numpy(), REFLECTION_BINS)
            ax.hist(get_target(self.num_pieces), REFLECTION_BINS)

        plt.axis("equal")

    def reflection_angle_distr(self) -> torch.tensor:
        _, outgoing_rays = self.reflect()

        return outgoing_rays[:, 2].flatten()

    def reflect(self) -> torch.tensor:
        dydx = torch.gradient(self.surface[:, 1], spacing=(self.surface[:, 0],))[0]
        mir_angles = torch.atan(dydx)

        incoming = self.surface
        dxs = incoming[:, 0]
        dys = self.source_h - incoming[:, 1]
        in_angles = TENSOR_PI - torch.atan(dys / dxs).view(-1, 1)
        incoming_rays = torch.hstack([incoming, in_angles])

        out_angles = TENSOR_PI - in_angles + 2 * mir_angles.view(-1, 1)
        outgoing_rays = torch.hstack([incoming, out_angles])

        return incoming_rays, outgoing_rays


def main():
    parser = argparse.ArgumentParser(description="Bulb Shade")
    parser.add_argument("-n", "--num-rays", type=int, default=32, help="Number of rays")

    args = parser.parse_args()

    loss = torch.nn.MSELoss()
    mirror = Mirror(args.num_rays, source_h=SOURCE_H)

    target = get_target(args.num_rays)
    optimizer = torch.optim.Adam([mirror._ys], lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=100
    )

    i = 0
    while True:

        vals = mirror.reflection_angle_distr()
        loss_ = loss(vals, target)
        if i % 1000 == 0:
            print(f"Loss: {loss_}")
        if i % 10000 == 0:
            mirror.plot()
            plt.show()
        loss_.backward()
        # mirror._ys.grad[0] = 0

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(loss_)

        i += 1


if __name__ == "__main__":
    main()

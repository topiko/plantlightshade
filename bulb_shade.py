import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

TENSOR_PI = torch.tensor(math.pi)
ALPHA = 0.1
NUM_BINS = 300
REFLECTION_BINS = torch.linspace(0, math.pi, NUM_BINS)


def cast(points_and_angles: np.ndarray, lens: np.ndarray) -> np.ndarray:
    points = points_and_angles[:, :2]
    angles = points_and_angles[:, 2]

    dxs = lens * np.cos(angles)
    dys = lens * np.sin(angles)

    return points + np.vstack([dxs, dys]).T


def get_target(num_rays: int) -> torch.tensor:
    return torch.linspace(TENSOR_PI / 2, TENSOR_PI / 7 * 3, num_rays, dtype=torch.float)


class Mirror:
    def __init__(self, num_rays: int, source_h: float, width: float = 1.0):
        num_pieces = num_rays
        self.num_pieces = num_pieces

        # TODO: starting from 0 causes divergence when taking the gradient.
        # Figure out why this is so.
        self._xs = torch.linspace(width / 1e12, width, num_pieces, requires_grad=False)
        self._ys = torch.zeros(num_pieces, requires_grad=True)
        self.source_h = source_h

    @property
    def surface(self) -> torch.tensor:
        return torch.vstack([self._xs, self._ys]).T

    def plot(self):
        surf = self.surface.detach().numpy()

        plt.plot(surf[:, 0], surf[:, 1], marker="o", markersize=2, color="black")

        incoming_rays, outgoing_rays = self.reflect()
        incoming_rays = incoming_rays.detach().numpy()
        outgoing_rays = outgoing_rays.detach().numpy()

        incoming_lens = (
            incoming_rays[:, 0] ** 2 + (self.source_h - incoming_rays[:, 1]) ** 2
        ) ** 0.5

        incoming_endpoints = cast(incoming_rays, incoming_lens)
        outgoing_endpoints = cast(outgoing_rays, incoming_lens)

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

        ax = plt.gca().inset_axes([0.0, 0.7, 0.4, 0.3])
        _common = {"bins": REFLECTION_BINS, "histtype": "step"}
        ax.hist(self.reflection_angle_distr().detach().numpy(), **_common)
        ax.hist(get_target(self.num_pieces), **_common)
        ax.set_xlabel("Angle [rad]")
        ax.set_title(
            "Reflection Angle Distribution",
            fontsize="x-small",
            loc="left",
            weight="bold",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.patch.set_alpha(0.5)

        plt.axis("off")
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
    parser.add_argument(
        "--shade-width", type=float, default=1.0, help="Width of the shade"
    )
    parser.add_argument(
        "--light-height", type=float, default=0.25, help="Height of the light source"
    )
    parser.add_argument(
        "--make-movie", action="store_true", help="Make figs for movie."
    )

    args = parser.parse_args()

    loss = torch.nn.MSELoss()
    mirror = Mirror(args.num_rays, source_h=args.light_height, width=args.shade_width)

    target = get_target(args.num_rays)
    optimizer = torch.optim.Adam([mirror._ys], lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=100
    )

    i = 0
    best_loss = float("inf")
    counter = 0
    while True:
        vals = mirror.reflection_angle_distr()
        loss_ = loss(vals, target)
        if i % 1000 == 0:
            print(f"Loss: {loss_}")
        if (i % 10 == 0) and args.make_movie:
            mirror.plot()
            plt.savefig(f"figs/{i:06d}.png")
            plt.clf()
        if i % 5000 == 0:
            mirror.plot()
            plt.show()

        loss_.backward()
        # Fix the origin of the mirror
        mirror._ys.grad[0] = 0

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(loss_)

        if loss_.item() < best_loss:
            best_loss = loss_.item()
            counter = 0

        if counter > 3e3:
            break
        i += 1
        counter += 1

    surf = mirror.surface.detach().numpy()

    np.savetxt(
        f"surface_w={args.shade_width}_source_h={args.light_height}.csv",
        surf,
        delimiter=",",
    )


if __name__ == "__main__":
    main()

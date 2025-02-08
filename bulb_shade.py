import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import cast, vecs_to_point

TENSOR_PI = torch.tensor(math.pi)
ALPHA = 0.1
NUM_BINS = 300
REFLECTION_BINS = torch.linspace(0, math.pi, NUM_BINS)


def get_target(num_rays: int, max_angle_deg: float = 0.0) -> torch.tensor:
    max_angle_rad = max_angle_deg / 180 * TENSOR_PI
    return torch.linspace(
        TENSOR_PI / 2, TENSOR_PI / 2 - max_angle_rad, num_rays, dtype=torch.float
    )


def loss(
    mirror: torch.tensor,
    outgoing_rays: torch.tensor,
    target_rays: torch.tensor,
    source_h: float,
    width: float,
) -> dict[str, torch.tensor]:
    mse = ((outgoing_rays - target_rays) ** 2).mean()

    # projections = get_projections(mirror, torch.tensor([0, source_h]))

    # we want all pieces to have same projected length:
    piece_lens = torch.diff(mirror, dim=0).norm(dim=1)
    piece_len_loss = ((piece_lens - 0.1) ** 2).sum()

    # width loss
    w_loss = (mirror[-1, 0] - width) ** 2

    # Angles
    vecs = vecs_to_point(mirror[1:], torch.tensor([0, source_h]))
    max_angle = TENSOR_PI / 4 * 3
    N = len(outgoing_rays)
    target_angles = torch.linspace(max_angle / N, max_angle, N) - TENSOR_PI / 2
    angles = torch.atan(vecs[:, 1] / vecs[:, 0])

    source_distr_loss = ((angles - target_angles) ** 2).sum()

    return {
        "mse": mse,
        "piece_lens": piece_len_loss,
        "w_loss": w_loss,
        "source_distr_loss": source_distr_loss,
    }


class Mirror:
    def __init__(
        self,
        num_rays: int,
        source_h: float,
        width: float = 1.0,
        max_angle_deg: float = 0.0,
    ):
        num_pieces = num_rays
        self.num_pieces = num_pieces

        self.width = width
        dx = width / (num_pieces - 1)
        # In self.surface we fix the origin to (0, 0).
        self._xs = torch.linspace(dx, width - dx, num_pieces - 2, requires_grad=True)
        self._ys = torch.zeros(num_pieces - 1, requires_grad=True)
        self._angles = torch.linspace(0, TENSOR_PI / 10, num_pieces, requires_grad=True)
        # self._angles = torch.zeros(num_pieces, requires_grad=True)
        self._lengths = (
            torch.ones(num_pieces, requires_grad=False) * dx
        ).requires_grad_()

        self.source_h = source_h
        self.max_angle_deg = max_angle_deg
        self.params = [self._angles, self._lengths]

    @property
    def surface(self) -> torch.tensor:

        dys = self._lengths * torch.sin(self._angles)
        dxs = self._lengths * torch.cos(self._angles)

        xs = torch.cumsum(dxs, dim=0)
        ys = torch.cumsum(dys, dim=0)
        xs = torch.concat((torch.tensor([0]), xs))
        ys = torch.concat((torch.tensor([0]), ys))
        return torch.vstack([xs, ys]).T

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
        ax.hist(
            get_target(self.num_pieces, max_angle_deg=self.max_angle_deg),
            **_common,
        )
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
        surf = self.surface
        dy = surf[1:, 1] - surf[:-1, 1]
        dx = surf[1:, 0] - surf[:-1, 0]

        dydx = dy / dx
        mir_angles = torch.atan(dydx)

        incoming = vecs_to_point(surf[1:], torch.tensor([0, self.source_h]))
        Dxs = incoming[:, 0]
        Dys = incoming[:, 1]

        in_angles = TENSOR_PI + torch.atan(Dys / Dxs).view(-1, 1)
        incoming_rays = torch.hstack([surf[1:], in_angles])

        out_angles = TENSOR_PI - in_angles + 2 * mir_angles.view(-1, 1)
        outgoing_rays = torch.hstack([surf[1:], out_angles])

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
        "--max-angle-deg",
        type=float,
        default=0.0,
        help="Maximum reflection angle in degrees (min = 0, straight up)",
    )
    parser.add_argument(
        "--make-movie", action="store_true", help="Make figs for movie."
    )
    parser.add_argument("--monitor", action="store_true", help="Monitor the training.")

    args = parser.parse_args()

    # loss = torch.nn.MSELoss()
    mirror = Mirror(
        args.num_rays,
        source_h=args.light_height,
        width=args.shade_width,
        max_angle_deg=args.max_angle_deg,
    )

    params = mirror.params
    target = get_target(args.num_rays, args.max_angle_deg)
    optimizer = torch.optim.Adam(params, lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.9, patience=1e3
    )

    loss_weights = {
        "mse": 1.0,
        "piece_lens": 0.0,
        "w_loss": 0.0,
        "source_distr_loss": 1.0,
    }
    i = 0
    best_loss = float("inf")
    counter = 0
    while True:
        ref_angles = mirror.reflection_angle_distr()
        loss_ = loss(
            mirror.surface, ref_angles, target, args.light_height, args.shade_width
        )

        # loss_weights["smooth_loss"] = loss_["mse"].item()
        loss_ = {k: v * loss_weights[k] for k, v in loss_.items()}
        loss_sum = sum(loss_.values())
        loss_sum.backward()
        if i % 100 == 0:
            print(counter)
            print(f"Loss={loss_sum.item():.3e}")
            for k, v in loss_.items():
                print(f"{k:>20}: {v.item():.3e}")

        if (i % 10 == 0) and args.make_movie:
            mirror.plot()
            plt.savefig(f"figs/{i:06d}.png")
            plt.clf()
        if args.monitor and (i % 1000 == 0):
            mirror.plot()
            plt.show()

            from utils import get_projections

            projs = get_projections(
                mirror.surface, torch.tensor([0, args.light_height])
            )
            norms = projs.norm(dim=1)
            plt.hist(norms.detach().numpy())
            plt.show()

        # torch.nn.utils.clip_grad_norm_(params, max_norm=1e-9)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(loss_sum)

        if loss_sum.item() < best_loss:
            best_loss = loss_sum.item()
            counter = 0

        if counter > 5e3:
            break
        i += 1
        counter += 1

    mirror.plot()
    plt.show()

    surf = mirror.surface.detach().numpy()

    np.savetxt(
        f"surface_w={args.shade_width}_source_h={args.light_height}_max_angle={args.max_angle_deg}.csv",
        surf,
        delimiter=",",
    )


if __name__ == "__main__":
    main()

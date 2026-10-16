"""Batched heightmap terrain: one (grid x grid) heightmap per rocket.

Heightmap index [n, i, j] holds the height at world x = -W/2 + i*cell, z = -W/2 + j*cell.
"""

import numpy as np

from .config import TerrainConfig


def _upsample(coarse: np.ndarray, size: int) -> np.ndarray:
    """Bilinearly upsample (M, c, c) grids to (M, size, size)."""
    c = coarse.shape[-1]
    t = np.linspace(0, c - 1, size)
    i0 = np.clip(np.floor(t).astype(int), 0, c - 2)
    f = t - i0
    # interpolate along axis 1, then axis 2
    a = coarse[:, i0, :] * (1 - f)[None, :, None] + coarse[:, i0 + 1, :] * f[None, :, None]
    return a[:, :, i0] * (1 - f)[None, None, :] + a[:, :, i0 + 1] * f[None, None, :]


class Terrain:
    def __init__(self, cfg: TerrainConfig, num_envs: int):
        self.cfg = cfg
        self.n = num_envs
        self.size = cfg.world_size
        self.cell = cfg.world_size / (cfg.grid - 1)
        self.heights = np.zeros((num_envs, cfg.grid, cfg.grid))
        self.pad = np.zeros((num_envs, 3))  # pad center (x, y, z)

    # ------------------------------------------------------------------ generation
    def regenerate(self, idx: np.ndarray, rng: np.random.Generator):
        """Build new terrain and pad locations for the envs in idx."""
        cfg = self.cfg
        m = len(idx)
        if m == 0:
            return
        g = cfg.grid
        h = np.zeros((m, g, g))
        if cfg.randomize:
            for o in range(cfg.octaves):
                c = 2 ** (o + 2) + 1
                h += _upsample(rng.standard_normal((m, c, c)), g) * (0.5 ** o)
            h *= cfg.amplitude / 2.0

        pad_xz = rng.uniform(-cfg.pad_range, cfg.pad_range, size=(m, 2))
        if not cfg.randomize:
            pad_xz[:] = 0.0

        # Flatten around the pad with a smoothstep blend toward the pad height.
        axis = np.linspace(-self.size / 2, self.size / 2, g)
        dx = axis[None, :, None] - pad_xz[:, 0, None, None]
        dz = axis[None, None, :] - pad_xz[:, 1, None, None]
        dist = np.sqrt(dx ** 2 + dz ** 2)
        self.heights[idx] = h
        pad_h = self.height_at(pad_xz[:, None, :], idx)[:, 0]
        r0, r1 = cfg.flatten_radius, cfg.flatten_radius * 2.0
        s = np.clip((dist - r0) / (r1 - r0), 0.0, 1.0)
        w = s * s * (3 - 2 * s)  # 0 inside the pad area, 1 far away
        self.heights[idx] = pad_h[:, None, None] * (1 - w) + h * w
        self.pad[idx] = np.stack([pad_xz[:, 0], pad_h, pad_xz[:, 1]], axis=1)

    # ------------------------------------------------------------------ queries
    def _bilinear(self, xz: np.ndarray, idx: np.ndarray | None):
        """xz: (M, K, 2). Returns heights (M, K) and gradient (M, K, 2) = (dh/dx, dh/dz)."""
        hm = self.heights if idx is None else self.heights[idx]
        g = self.cfg.grid
        u = (xz[..., 0] + self.size / 2) / self.cell
        v = (xz[..., 1] + self.size / 2) / self.cell
        u = np.clip(u, 0, g - 1 - 1e-9)
        v = np.clip(v, 0, g - 1 - 1e-9)
        i0 = np.minimum(np.floor(u).astype(int), g - 2)
        j0 = np.minimum(np.floor(v).astype(int), g - 2)
        fu, fv = u - i0, v - j0
        n = np.arange(hm.shape[0])[:, None]
        h00 = hm[n, i0, j0]
        h10 = hm[n, i0 + 1, j0]
        h01 = hm[n, i0, j0 + 1]
        h11 = hm[n, i0 + 1, j0 + 1]
        h = (h00 * (1 - fu) * (1 - fv) + h10 * fu * (1 - fv)
             + h01 * (1 - fu) * fv + h11 * fu * fv)
        dhdx = ((h10 - h00) * (1 - fv) + (h11 - h01) * fv) / self.cell
        dhdz = ((h01 - h00) * (1 - fu) + (h11 - h10) * fu) / self.cell
        return h, np.stack([dhdx, dhdz], axis=-1)

    def height_at(self, xz: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        return self._bilinear(xz, idx)[0]

    def height_and_normal(self, xz: np.ndarray, idx: np.ndarray | None = None):
        """Returns heights (M, K) and unit surface normals (M, K, 3)."""
        h, grad = self._bilinear(xz, idx)
        nrm = np.stack([-grad[..., 0], np.ones_like(h), -grad[..., 1]], axis=-1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        return h, nrm

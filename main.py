import numpy as np
import pandas as pd
import plotly.express as px
import requests
from ase import units
from ase.build import fcc111, molecule
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import LBFGS
from mace.calculators import mace_mp
from pydantic import Field
from zndraw import ZnDraw
from zndraw.extensions import Category, Extension

model = mace_mp()


class MolecularDynamics(Extension):
    category = Category.MODIFIER

    velocity: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Initial velocity of water toward slab (Å/fs)",
        json_schema_extra={"format": "range", "min": 0.0, "max": 0.5, "step": 0.01},
    )
    timestep: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="MD timestep (fs)",
        json_schema_extra={"format": "range", "min": 0.1, "max": 5.0, "step": 0.1},
    )
    steps: int = Field(
        default=200,
        ge=10,
        le=1000,
        description="Number of MD steps",
        json_schema_extra={"format": "range", "min": 10, "max": 1000, "step": 10},
    )
    temperature: float = Field(
        default=300.0,
        ge=0.0,
        le=500.0,
        description="Initial temperature for slab atoms (K)",
        json_schema_extra={"format": "range", "min": 0.0, "max": 500.0, "step": 10.0},
    )

    def run(self, vis: ZnDraw, **kwargs):
        vis.step = 0
        del vis[1:]

        # Get the current atoms and attach calculator
        atoms = vis.atoms
        atoms.calc = model
        atoms.info.pop("connectivity", None)

        # Find water molecule indices (last 3 atoms: O, H, H)
        n_atoms = len(atoms)
        water_indices = list(range(n_atoms - 3, n_atoms))
        slab_indices = list(range(n_atoms - 3))

        # Get slab top z-coordinate for distance calculation
        slab_top_z = atoms.positions[slab_indices, 2].max()

        # Initialize velocities with Maxwell-Boltzmann distribution
        if self.temperature > 0:
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)

        # Set directed velocity on water molecule toward surface (negative z)
        velocities = atoms.get_velocities()
        water_velocity = np.array([0.0, 0.0, -self.velocity])
        for idx in water_indices:
            velocities[idx] = water_velocity
        atoms.set_velocities(velocities)

        # Run MD simulation
        dyn = VelocityVerlet(atoms, timestep=self.timestep * units.fs)

        # Track distance and energy
        distances = []
        energies = []

        with vis.progress_tracker(f"Running MD ({self.steps} steps)") as tracker:
            for step in range(self.steps):
                dyn.run(1)
                vis.append(atoms.copy())
                vis.step += 1

                # Calculate distance (water oxygen z - slab top z)
                water_z = atoms.positions[water_indices[0], 2]  # Oxygen atom
                distance = water_z - slab_top_z
                distances.append(distance)

                # Get potential energy
                energy = atoms.get_potential_energy()
                energies.append(energy)

                # Update figures every 10 steps
                if (step + 1) % 10 == 0 or step == self.steps - 1:
                    self._update_figures(vis, distances, energies)

                tracker.update(
                    f"Step {step + 1}/{self.steps}",
                    progress=(step + 1) / self.steps * 100,
                )

    def _update_figures(self, vis: ZnDraw, distances: list, energies: list):
        steps = list(range(len(distances)))
        meta_step = np.arange(len(distances))

        # Distance figure
        df_dist = pd.DataFrame({"step": steps, "distance": distances})
        fig_dist = px.scatter(
            df_dist,
            x="step",
            y="distance",
            labels={"step": "Frame", "distance": "Distance (Å)"},
            title="Water-Surface Distance",
        )
        fig_dist.add_scatter(
            x=df_dist["step"],
            y=df_dist["distance"],
            mode="lines",
            name="trend",
            line=dict(color="rgba(0, 0, 0, 0.1)"),
            hoverinfo="skip",
            showlegend=False,
        )
        fig_dist.update_traces(
            customdata=np.stack([meta_step], axis=-1),
            selector=dict(mode="markers"),
            meta={
                "interactions": [{"click": "step", "select": "step", "hover": "step"}]
            },
        )
        fig_dist.update_layout(dragmode="lasso", hovermode="closest")
        vis.figures["Distance"] = fig_dist

        # Energy figure
        df_energy = pd.DataFrame({"step": steps, "energy": energies})
        fig_energy = px.scatter(
            df_energy,
            x="step",
            y="energy",
            labels={"step": "Frame", "energy": "Energy (eV)"},
            title="Potential Energy",
        )
        fig_energy.add_scatter(
            x=df_energy["step"],
            y=df_energy["energy"],
            mode="lines",
            name="trend",
            line=dict(color="rgba(0, 0, 0, 0.1)"),
            hoverinfo="skip",
            showlegend=False,
        )
        fig_energy.update_traces(
            customdata=np.stack([meta_step], axis=-1),
            selector=dict(mode="markers"),
            meta={
                "interactions": [{"click": "step", "select": "step", "hover": "step"}]
            },
        )
        fig_energy.update_layout(dragmode="lasso", hovermode="closest")
        vis.figures["Energy"] = fig_energy


def main():
    server_url = "http://localhost:4567"
    room = "demo-room"

    vis = ZnDraw(url=f"{server_url}/", room=room, user="user-ba91fc6b")

    # Set this room as the default room to extend from
    headers = vis.api._get_headers()
    requests.put(
        f"{server_url}/api/rooms/default",
        json={"roomId": room},
        headers=headers,
    ).raise_for_status()

    # Create and optimize water molecule
    print("Optimizing water molecule...")
    water = molecule("H2O")
    water.calc = model
    opt_water = LBFGS(water)
    opt_water.run(fmax=0.01)

    # Create gold (111) surface slab
    print("Optimizing gold surface...")
    slab = fcc111("Au", size=(4, 4, 3), vacuum=10.0)
    slab.calc = model

    # Fix bottom layer atoms during optimization
    bottom_z = slab.positions[:, 2].min()
    fixed_indices = [
        i for i, pos in enumerate(slab.positions) if pos[2] < bottom_z + 0.5
    ]
    slab.set_constraint(FixAtoms(indices=fixed_indices))

    opt_slab = LBFGS(slab)
    opt_slab.run(fmax=0.05)

    # Remove constraint for MD
    slab.set_constraint()

    # Position water above the optimized surface
    slab_top = slab.positions[:, 2].max()
    water_distance = 5.0  # Angstroms above the surface

    # Center water over the slab
    slab_center = np.diag(slab.cell)[:2] / 2
    water.translate([slab_center[0], slab_center[1], slab_top + water_distance])

    # Combine slab and water
    atoms = slab + water

    print("Setup complete. Starting ZnDraw...")
    vis.append(atoms)

    # Register the MD extension
    vis.register_extension(MolecularDynamics, public=True)
    vis.wait()


if __name__ == "__main__":
    main()

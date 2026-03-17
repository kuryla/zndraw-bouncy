import numpy as np
import pandas as pd
import plotly.express as px
import requests
from ase import units
import os
import ase
from ase.io import read, write
from ase.build import fcc111, molecule
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import LBFGS
from mace.calculators import mace_mp
from pydantic import Field
from zndraw import ZnDraw
from zndraw.extensions import Category, Extension
from pathlib import Path

model = mace_mp("small", device="cuda", default_dtype="float32")
structs_path = Path("structures/graphene-c60")
structs_path.mkdir(exist_ok=True, parents=True)

class MolecularDynamics(Extension):
    category = Category.MODIFIER

    velocity: float = Field(
        default=200,
        ge=0.0,
        le=5000.0,
        description="Initial velocity of water toward slab (m/s)",
        json_schema_extra={"format": "range", "min": 0.0, "max": 5000.0, "step": 200},
    )

    def run(self, vis: ZnDraw, **kwargs):
        steps = 2000
        timestep = 1 * units.fs
        temperature = 300
        vis.step = 0
        del vis[1:]

        if os.path.exists(structs_path / f"{self.velocity:.1f}.xyz"):
            vis.extend(read(structs_path / f"{self.velocity:.1f}.xyz", ":"))
        else:
            # run the MD, append to vis and in parallel write the xyz file or at the end do ase.io.write(..., list(vis))

            # Get the current atoms and attach calculator
            atoms = vis.atoms
            atoms.calc = model
            atoms.info.pop("connectivity", None)

            # Find water molecule indices (last 3 atoms: O, H, H)
            n_atoms = len(atoms)


            # Initialize velocities with Maxwell-Boltzmann distribution
            if temperature > 0:
                MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

            # Set directed velocity on water molecule toward surface (negative z)
            velocities = atoms.get_velocities()
            added_velocity = np.array([0.0, 0.0, -self.velocity])
            for idx, _ in enumerate(atoms):
                if atoms.arrays["velocity_mask"][idx]:
                    velocities[idx] += added_velocity
            atoms.set_velocities(velocities)

            # Run MD simulation
            dyn = VelocityVerlet(atoms, timestep=timestep)

            # Track distance and energy
            distances = []
            energies = []

            with vis.progress_tracker(f"Running MD ({steps} steps)") as tracker:
                for step in range(steps):
                    dyn.run(1)
                    vis.append(atoms.copy())
                    vis.step += 1

                    distances.append(0)

                    # Get potential energy
                    energy = atoms.get_potential_energy()
                    energies.append(energy)

                    # Update figures every 10 steps
                    if (step + 1) % 10 == 0 or step == steps - 1:
                        self._update_figures(vis, distances, energies)

                    tracker.update(
                        f"Step {step + 1}/{steps}",
                        progress=(step + 1) / steps * 100,
                    )
            write(structs_path / f"/{self.velocity:.1f}.xyz", list(vis))
            

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
            title="Graphene-Fullerene Distance",
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
    room = "Graphene and fullerene"

    vis = ZnDraw(url=f"{server_url}/", room=room, user="user-ba91fc6b")

    # Set this room as the default room to extend from
    headers = vis.api._get_headers()
    requests.put(
        f"{server_url}/api/rooms/default",
        json={"roomId": room},
        headers=headers,
    ).raise_for_status()

    # Create and optimize water molecule
    print("Optimizing fullerene...")
    fullerene = read("structures/C60.xyz")
    fullerene.calc = model
    opt_fullerene = LBFGS(fullerene)
    opt_fullerene.run(fmax=0.1)

    # Read graphene sheet
    print("Optimizing graphene surface...")
    slab = read("structures/graphene.xyz")
    slab.calc = model

    # Fix bottom layer atoms during optimization

    opt_slab = LBFGS(slab)
    opt_slab.run(fmax=0.1)

    # Remove constraint for MD
    slab.set_constraint()
    """
    # Position water above the optimized surface
    slab_top = slab.positions[:, 2].max()
    water_distance = 5.0  # Angstroms above the surface

    # Center water over the slab
    slab_center = np.diag(slab.cell)[:2] / 2
    water.translate([slab_center[0], slab_center[1], slab_top + water_distance])
    """

    # Combine slab and water

    slab.arrays["velocity_mask"] = np.full(len(slab), False)
    fullerene.arrays["velocity_mask"] = np.full(len(fullerene), True)
    atoms = slab + fullerene

    print("Setup complete. Starting ZnDraw...")
    vis.append(atoms)

    # Register the MD extension
    vis.register_extension(MolecularDynamics, public=True)
    vis.wait()


if __name__ == "__main__":
    main()

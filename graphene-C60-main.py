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
from pydantic import Field
from zndraw import ZnDraw
from zndraw.extensions import Category, Extension
from pathlib import Path

structs_path = Path("../structures/graphene-C60")
structs_path.mkdir(exist_ok=True, parents=True)

class MolecularDynamics(Extension):
    category = Category.MODIFIER

    velocity: float = Field(
        default=200,
        ge=200.0,
        le=4000.0,
        description="Initial velocity of water toward slab (m/s)",
        json_schema_extra={"format": "range", "min": 200.0, "max": 4000.0, "step": 200},
    )

    def run(self, vis: ZnDraw, **kwargs):
        steps = 2000
        timestep = 1 * units.fs
        temperature = 300
        vis.step = 0
        del vis[1:]

        atoms_list = read(structs_path / f"{self.velocity:.0f}.xyz", ":")

        vis.extend(atoms_list)


def main():
    server_url = "http://localhost:4567"
    room = "Graphene-fullerene"

    vis = ZnDraw(url=f"{server_url}/", room=room, user="user-ba91fc6b")

    # Set this room as the default room to extend from
    headers = vis.api._get_headers()
    requests.put(
        f"{server_url}/api/rooms/default",
        json={"roomId": room},
        headers=headers,
    ).raise_for_status()

    atoms = read("../structures/graphene-C60.xyz")
    vis.append(atoms)
    if "cell" in vis.geometries:
        del vis.geometries["cell"]

    # Register the MD extension
    vis.register_extension(MolecularDynamics, public=True)
    vis.wait()


if __name__ == "__main__":
    main()

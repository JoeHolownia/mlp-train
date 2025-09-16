"""Useful tools for optimisation, solvation etc. using MLPs."""
from mlptrain.sampling.md import _convert_ase_traj
from mlptrain.potentials import MLPotential
from mlptrain.configurations import Configuration
from mlptrain.config import Config
from mlptrain.utils import work_in_tmp_dir as work_in_tmp_dir_mlt
import os
from mlptrain.log import logger


# @work_in_tmp_dir_mlt()
def optimise_with_fix_solute(
    config: Configuration,
    mlp: MLPotential,
    solute: Configuration = None,
    fmax: float = 0.01,
    **kwargs,
) -> Configuration:
    """
    Optimise the configuration by MLP with a fixed solute (solute coords should be the first in configuration coords).

    Parameters:
        config (mlt.Configuration): the configuration either in vacuum or in solvent where the first len(solute) atoms
                                    are those of the solute.
        mlp (mlt.potentials.MLPotential):
        solute (mlt.Configuration): 'solute' configuration, if specified, takes the number of atoms in this config to determine
                                    the first n atoms of 'config' to fix with constraints.
        fmax (float):               fmax value for BFGS optimiser

    Returns:
        mlt.Configuration: final frame config of optimised trajectory.
    """
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS
    from ase.io.trajectory import Trajectory as ASETrajectory

    assert config.box is not None, 'configuration must have box'
    logger.info(
        'Optimise the configuration with fixed solute (solute coords should at the first in configuration coords) by MLP'
    )

    n_cores = (
        kwargs['n_cores'] if 'n_cores' in kwargs else min(Config.n_cores, 8)
    )
    os.environ['OMP_NUM_THREADS'] = str(n_cores)
    logger.info(f'Using {n_cores} cores for MLP MD')

    # get ase atoms and load calculator
    ase_atoms = config.ase_atoms
    logger.info(f'{ase_atoms.cell}, {ase_atoms.pbc}')
    ase_atoms.calc = mlp.ase_calculator

    # constrain solute atoms if specified
    if solute is not None:
        solute_idx = list(range(len(solute.atoms)))
        constraints = FixAtoms(indices=solute_idx)
        ase_atoms.set_constraint(constraints)

    # run optimisation
    asetraj = ASETrajectory('tmp.traj', 'w', ase_atoms)
    dyn = BFGS(ase_atoms)
    dyn.attach(asetraj.write, interval=2)
    dyn.run(fmax=fmax)

    # return final optimisation trajectory frame
    traj = _convert_ase_traj('tmp.traj')
    final_traj = traj.final_frame
    return final_traj


def mlp_ts_opt_hessian():



    return
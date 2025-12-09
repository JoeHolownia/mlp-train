import os
import mlptrain as mlt
import torch
from ase.io.trajectory import Trajectory as ASETrajectory
from .data.utils import work_in_zipped_dir
from mlptrain.config import Config
import pytest

# Config.n_cores = 1
# Config.mace_device = 'cuda'
# Config.mace_params['cueq'] = False

# @pytest.fixture
# def set_gpu_config():
#     """Fixture to set required config values for ORCA and Gaussian keywords"""

#     # 1. save the current values
#     # n_cores_kws = Config.n_cores
#     # mace_device_kws = Config.mace_device
#     # mace_params_kws = Config.mace_params

#     # 2. patch variables with temporary test-specific values
#     Config.n_cores = 1
#     Config.mace_device = 'cuda'
#     Config.mace_params['cueq'] = False

#     # 3. run test
#     # yield

#     # # 4. restore variables
#     # Config.n_cores = n_cores_kws
#     # Config.mace_device = mace_device_kws
#     # Config.mace_params = mace_params_kws

# pytest flag to control only locally running tests
localonly = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Currently skipping GPU tests on GitHub Actions - test these locally."
)

here = os.path.abspath(os.path.dirname(__file__))

@localonly
@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
# @pytest.mark.parametrize("cueq", [False])
def test_mace_gpu_cueq(h2o_system_config, monkeypatch):
    """Test MACE MD run with gpu + cuequivariance acceleration. """

    Config.n_cores = 1
    Config.mace_device = 'cuda'
    Config.mace_params['cueq'] = False

    monkeypatch.setattr('mlptrain.potentials.mace.mace.Config', Config)
    monkeypatch.setattr('mlptrain.md.Config', Config)

    assert Config.n_cores == 1
    assert Config.mace_device == 'cuda'
    assert Config.mace_params['device'] == 'cuda'
    assert torch.cuda.is_available()

    # Check the default device
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"CUDA Version: {torch.version.cuda}")

    # H2O molecule
    system, config = h2o_system_config

    mace = mlt.potentials.MACE('water', system=system, foundation='medium_off')

    # run some dynamics with the potential
    mlt.md.run_mlp_md(
            configuration=config,
            mlp=mace,
            temp=300,
            dt=1,
            interval=10,
            kept_substrings=['.traj'],
            ps=0.5
    )

    assert os.path.exists('trajectory.traj')
    traj = ASETrajectory('trajectory.traj')

    # 500 fs / 10 + 1 starting frame
    assert len(traj) == 50 + 1

import numpy as np

from backend.app.core.candidates import FEATURE_NAMES, build_candidates, make_observations
from backend.app.core.hmm import viterbi_decode
from backend.app.core.network import RoadNetwork
from backend.app.core.trip_io import TripInput, load_trip_input
from backend.tests.conftest import _write_trip


def test_viterbi_prefers_consistent_path_over_noisy_emission():
    # 2 states per step. Emission favours state 1 at t=1 (noise) but transitions forbid switching.
    emission = [np.log([0.9, 0.1]), np.log([0.4, 0.6]), np.log([0.9, 0.1])]
    stay = np.array([[0.0, -10.0], [-10.0, 0.0]])
    assert viterbi_decode(emission, lambda t: stay).tolist() == [0, 0, 0]


def test_viterbi_restart_breaks_dependency():
    emission = [np.log([0.9, 0.1]), np.log([0.1, 0.9])]
    stay = np.array([[0.0, -50.0], [-50.0, 0.0]])
    assert viterbi_decode(emission, lambda t: stay).tolist() == [0, 0]
    assert viterbi_decode(emission, lambda t: stay, np.array([False, True])).tolist() == [0, 1]


def test_candidates_every_observation_has_one_and_contain_truth(tmp_path):
    route = _write_trip(tmp_path / "s", "00000001")
    trip = load_trip_input(tmp_path / "s", "00000001")
    net = RoadNetwork.from_trip(trip)
    obs = make_observations(trip, net)
    c = build_candidates(obs, net, radius_m=75, max_candidates=8)
    assert (np.diff(c.offsets) >= 1).all()
    assert c.X.shape[1] == len(FEATURE_NAMES) and np.isfinite(c.X).all()
    assert np.isin(c.arc_id, np.arange(net.n_arcs)).all()


def test_reverse_twin_and_connectivity(tmp_path):
    _write_trip(tmp_path / "s", "00000001")
    net = RoadNetwork.from_trip(load_trip_input(tmp_path / "s", "00000001"))
    a = 0
    twin = net.reverse_of[a]
    assert twin >= 0 and net.arc_from[twin] == net.arc_to[a] and net.arc_to[twin] == net.arc_from[a]
    nxt = int(net.successors(a)[0])
    assert net.connected(a, nxt)


def test_trip_input_cannot_carry_labels(tmp_path):
    names = set(TripInput.__dataclass_fields__)
    assert not any("route" in n or "label" in n or "truth" in n for n in names)
    _write_trip(tmp_path / "s", "00000001", with_route=False)       # loader must work with NO .route file
    assert load_trip_input(tmp_path / "s", "00000001").n_obs > 0

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from backend.app.core.candidates import FEATURE_NAMES


def test_health_and_trips(make_client):
    with make_client() as c:
        h = c.get("/health").json()
        assert h["status"] == "ok" and h["trip_set"] == "student_test" and h["model_loaded"] is False
        t = c.get("/trips").json()
        assert [x["trip_id"] for x in t] == ["00000001"]
        assert len(c.get("/trips/00000001").json()["lat"]) == t[0]["n_observations"]
        assert c.get("/trips/00000001/network").json()["n_arcs"] > 0


def test_predict_baseline_plus_hmm_is_accurate_and_ground_truth_stays_server_side(make_client):
    with make_client() as c:
        r = c.post("/predict", json={"trip_id": "00000001"})
        assert r.status_code == 200
        body = r.json()
        assert body["summary"]["top1_accuracy"] > 0.9
        assert body["metrics"]["hmm"]["route_continuity"] > 0.95
        assert body["reference_route_geometry"] is None              # not exposed by default
        assert len(body["predicted_arc_per_observation"]) == body["n_observations"]


def test_reference_geometry_only_when_explicitly_enabled(make_client):
    with make_client(EXPOSE_REFERENCE_ROUTE="true") as c:
        assert c.post("/predict", json={"trip_id": "00000001"}).json()["reference_route_geometry"]


def test_predictions_do_not_depend_on_labels(make_client, data_root):
    with make_client() as c:
        with_labels = c.post("/predict", json={"trip_id": "00000001"}).json()
    (data_root / "student_test/00000001/00000001.route").unlink()
    with make_client() as c:
        without = c.post("/predict", json={"trip_id": "00000001"}).json()
    assert without["metrics"] is None and without["summary"] is None
    assert without["predicted_arc_per_observation"] == with_labels["predicted_arc_per_observation"]


def test_bad_inputs(make_client):
    with make_client() as c:
        assert c.post("/predict", json={"trip_id": "00000099"}).status_code == 404
        assert c.post("/predict", json={"trip_id": "../../etc/passwd"}).status_code == 422
        assert c.post("/predict", json={"trip_id": "00000001", "matcher": "nope"}).status_code == 400
        assert c.get("/trips/00000099").status_code == 404


def _fit_bundle(path, names=FEATURE_NAMES):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(names)))
    m = LogisticRegression().fit(X, (X[:, 0] < 0).astype(int))   # arbitrary: just needs predict_proba
    joblib.dump({"model": m, "feature_names": list(names), "name": "dummy_lr"}, path)


def test_model_bundle_is_loaded_and_used(make_client, tmp_path):
    art = tmp_path / "art"; art.mkdir(); _fit_bundle(art / "model.joblib")
    with make_client(artifacts=art) as c:
        assert c.get("/health").json()["model_loaded"] is True
        body = c.post("/predict", json={"trip_id": "00000001"}).json()
        assert body["matcher"] == "dummy_lr"
        assert body["metrics"]["candidates"]["log_loss_pairwise"] is not None
        assert c.post("/predict", json={"trip_id": "00000001", "matcher": "nearest_road"}).json()["matcher"] == "nearest_road"


def test_model_with_unknown_features_is_rejected(make_client, tmp_path):
    art = tmp_path / "art"; art.mkdir(); _fit_bundle(art / "model.joblib", names=["dist_m", "bogus_feature"])
    with pytest.raises(ValueError, match="unknown to the backend"):
        with make_client(artifacts=art):
            pass


def test_require_model_refuses_to_start_without_one(make_client):
    with pytest.raises(RuntimeError, match="model required"):
        with make_client(REQUIRE_MODEL="true"):
            pass

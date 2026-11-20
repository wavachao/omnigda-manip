import numpy as np

from omnigda.core.gda_math import (
    OnlineShrinkageGDA,
    backproject_depth,
    entropy_gated_bayesian_fusion,
    estimate_shrunk_covariance,
    invert_spd,
    woodbury_precision_update,
)


def test_shrinkage_covariance_spd_when_n_less_than_d():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 128))
    x -= x.mean(axis=0, keepdims=True)
    for method in ("oas", "ledoit_wolf", "ridge"):
        cov, alpha = estimate_shrunk_covariance(x, method=method)
        assert cov.shape == (128, 128)
        assert 0.0 <= alpha <= 1.0
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0)


def test_woodbury_matches_exact_inverse():
    rng = np.random.default_rng(7)
    d, k = 64, 4
    a = rng.normal(size=(d, d))
    sigma = a @ a.T / d + 0.5 * np.eye(d)
    prec = invert_spd(sigma)
    u = rng.normal(size=(d, k))
    scale = 0.2
    prec_woodbury = woodbury_precision_update(prec, u, scale=scale)
    prec_exact = invert_spd(sigma + scale * (u @ u.T))
    assert np.allclose(prec_woodbury, prec_exact, atol=1e-7)


def test_online_shrinkage_gda_registration_and_ood():
    rng = np.random.default_rng(42)
    d = 64
    c0 = rng.normal(loc=0.0, scale=0.1, size=(12, d))
    c0[:, 0] += 3.0
    c1 = rng.normal(loc=0.0, scale=0.1, size=(12, d))
    c1[:, 1] += 3.0
    gda = OnlineShrinkageGDA(shrinkage_method="oas")
    gda.fit({"mug": c0, "bottle": c1})

    # Register a new class online (few-shot)
    c2 = rng.normal(loc=0.0, scale=0.1, size=(3, d))
    c2[:, 2] += 3.0
    gda.register_online("wrench", c2, use_woodbury=True)
    assert gda.class_order == ["mug", "bottle", "wrench"]

    q = np.zeros((3, d))
    q[0, 0] = 3.0
    q[1, 1] = 3.0
    q[2, 2] = 3.0
    probs = gda.predict_proba(q)
    assert np.array_equal(np.argmax(probs, axis=1), np.array([0, 1, 2]))

    # OOD sample far away from all prototypes should have higher Mahalanobis distance
    ood = np.zeros((1, d))
    ood[0, 10] = 3.0
    d_in = np.min(gda.mahalanobis_sq(q[:1]))
    d_ood = np.min(gda.mahalanobis_sq(ood))
    assert d_ood > d_in * 2.0


def test_entropy_gated_bayesian_fusion_prefers_confident_modality():
    # Modality 2D is ambiguous (near-uniform), Modality 3D is confident on class 1
    ll_2d = np.array([0.05, -0.05, 0.0])
    ll_3d = np.array([-3.0, 5.0, -3.0])
    probs, ent, weights = entropy_gated_bayesian_fusion(
        {"2d": ll_2d, "3d": ll_3d}, reliability={"2d": 1.0, "3d": 1.0}
    )
    assert weights["3d"] > weights["2d"]
    assert int(np.argmax(probs)) == 1
    assert 0.0 <= ent <= 1.0


def test_backproject_depth():
    depth = np.ones((10, 10), dtype=np.float32) * 1.0
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    pts, _ = backproject_depth(depth, fx=100.0, fy=100.0, cx=5.0, cy=5.0, mask=mask)
    assert pts.shape == (1, 3)
    assert np.allclose(pts[0], [0.0, 0.0, 1.0], atol=1e-5)

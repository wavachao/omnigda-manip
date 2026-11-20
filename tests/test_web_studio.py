"""Integration tests for OmniGDA-Manip Interactive Web Studio."""

from __future__ import annotations

from omnigda.web.server import OmniGDAStudioBackend, create_app


def test_web_studio_endpoints_end_to_end():
    studio = OmniGDAStudioBackend(backend="tensorrt", seed=42, num_objects=5, include_novel=True)
    try:
        app = create_app(studio)
        client = app.test_client()

        resp = client.get("/")
        assert resp.status_code == 200
        assert b"OmniGDA-Manip" in resp.data

        st = client.get("/api/state").get_json()
        assert st["backend"] == "tensorrt"
        assert len(st["annotated_jpg"]) > 1000
        assert len(st["scene_objects"]) >= 5

        sw = client.post("/api/switch_camera", json={}).get_json()
        assert sw["camera"] == "nbv_camera"
        sw2 = client.post("/api/switch_camera", json={}).get_json()
        assert sw2["camera"] == "base_camera"

        reg = client.post("/api/register_preset", json={"group": "novel", "num_shots": 2}).get_json()
        assert "cyan_t_tool" in reg["registered_classes"]
        assert reg["registration_ms"] > 0.0

        perc = client.post("/api/perceive", json={}).get_json()
        assert len(perc["detections"]) >= 1
        assert len(perc["heatmap_jpg"]) > 1000
        assert perc["metrics"]["seg2d_ms"] > 0.0
    finally:
        studio.sim.close()

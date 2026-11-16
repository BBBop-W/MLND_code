"""Contract checks for validated, offline server map rendering."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import server_plot


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


class ServerMapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.instance = self.root / "instance"
        self.instance.mkdir()
        self.run = self.root / "run"
        self.run.mkdir()
        self.maps = self.root / "maps"
        self.maps.mkdir()
        (self.root / "provenance").mkdir()
        self.metadata = {"projection_center_lon": 0, "projection_center_lat": 0,
                         "crop_bounds_local_km": {"x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1}}
        write_json(self.instance / "metadata.json", self.metadata)
        write_json(self.instance / "model.json", {"ids": {"bases": ["L0"], "hubs": ["H0", "H1"],
                                                          "flight_nodes": ["L0", "H0", "H1"]}})
        (self.instance / "sites.csv").write_text("site_id,type,x_km,y_km\nL0,L,0,0\nH0,H,0.1,0\nH1,H,0.2,0.1\n")
        self.manifest = {"files": {name: server_plot.sha256(self.instance / name)
                                   for name in ("model.json", "metadata.json", "sites.csv")}}
        write_json(self.instance / "manifest.json", self.manifest)
        write_json(self.run / "result.json", {"assignment_scope": "task", "demand_formulation": "preprocessed",
                                              "delivery_time_scope": "courier_home", "max_orders": 2})
        write_json(self.run / "solution.json", {"facility": [1, 1, 0],
                                                "airline": [[0, 1, 0], [1, 0, 0], [0, 0, 0]]})
        write_json(self.run / "input_reference.json", {"max_orders": 2, "manifest": self.manifest})
        # Repeated service legs must each count; shared segment witnesses must
        # not be counted once more as operating flow.
        write_json(self.run / "path_flows.json", {
            "missions": [{"flow": 2.0, "path": {"legs": [[0, 1], [1, 0], [0, 1]]}}],
            "relocations": [{"flow": 0.5, "path": {"legs": [[1, 0]]}}],
            "segments": [{"flow": 999.0, "legs": [[0, 1]]}]})
        self.certify()

    def certify(self):
        certificate = {"passed": True, "checks": {"passed": True}, "native_manifest_matches": True,
                       "validation_scope": "current_model_primal_and_complete_input_preprocessing",
                       "assignment_scope": "task", "delivery_time_scope": "courier_home", "max_orders": 2,
                       "manifest_sha256": server_plot.sha256(self.instance / "manifest.json")}
        certificate.update({name + "_sha256": server_plot.sha256(self.run / (name + ".json"))
                            for name in ("result", "solution", "path_flows", "input_reference")})
        write_json(self.run / "native_validation.json", certificate)

    def test_operating_arcs_count_traversals_without_segment_double_counting(self):
        state = server_plot.load_verified_run(server_plot.load_input(self.instance), self.run)
        self.assertEqual(state["arcs"], {(0, 1): [4.0, 0.0], (1, 0): [2.0, 0.5]})

    def test_changed_native_output_is_rejected(self):
        (self.run / "solution.json").write_text('{"facility": [0, 0, 0]}')
        with self.assertRaisesRegex(ValueError, "no longer matches solution"):
            server_plot.load_verified_run(server_plot.load_input(self.instance), self.run)

    def test_old_or_incomplete_certificate_is_rejected(self):
        path = self.run / "native_validation.json"
        certificate = server_plot.read_json(path)
        certificate["validation_scope"] = "old_partial_witness_check"
        write_json(path, certificate)
        with self.assertRaisesRegex(ValueError, "validation did not pass"):
            server_plot.load_verified_run(server_plot.load_input(self.instance), self.run)

    def test_positive_flow_cannot_use_closed_airline(self):
        path = self.run / "solution.json"
        solution = server_plot.read_json(path)
        solution["airline"][0][1] = 0
        write_json(path, solution)
        self.certify()
        with self.assertRaisesRegex(ValueError, "closed airline"):
            server_plot.load_verified_run(server_plot.load_input(self.instance), self.run)

    def test_input_tampering_is_rejected(self):
        (self.instance / "sites.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Input SHA-256 mismatch"):
            server_plot.load_input(self.instance)

    def test_input_manifest_cannot_escape_instance(self):
        (self.root / "outside.txt").write_text("outside")
        self.manifest["files"]["../outside.txt"] = server_plot.sha256(self.root / "outside.txt")
        write_json(self.instance / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "outside-root"):
            server_plot.load_input(self.instance)

    def test_background_tampering_is_rejected(self):
        background = self.maps / "background.png"
        background.write_bytes(b"test placeholder; not rendered")
        entry = {**self.metadata, "file": "maps/background.png", "extent_epsg3857": [-1000, 1000, -1000, 1000],
                 "sha256": server_plot.sha256(background)}
        write_json(self.root / "provenance/map_backgrounds.json", {"backgrounds": [entry]})
        background.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "background SHA-256 mismatch"):
            server_plot.select_background(server_plot.load_input(self.instance), self.maps)


if __name__ == "__main__":
    unittest.main()

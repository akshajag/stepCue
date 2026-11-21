import json
import os
import ssl
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import companion_server as srv  # noqa: E402
import companion_store as cs  # noqa: E402
import location_io  # noqa: E402

KEY = "test-key-123"


class Api(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        comp = os.path.join(self.tmp.name, "companion.json")
        loc = os.path.join(self.tmp.name, "location.json")
        for target, attr, val in ((cs, "COMPANION_FILE", comp), (srv, "LOCATION_FILE", loc),
                                  (location_io, "LOCATION_FILE", loc)):
            p = mock.patch.object(target, attr, val)
            p.start()
            self.addCleanup(p.stop)
        self.loc = loc

    def test_checkin_confirm_phone_health(self):
        self.assertEqual(srv.handle_api("/api/checkin", {"kind": "help"})[0], 200)
        self.assertEqual(srv.handle_api("/api/confirm", {"episode_id": 4, "froze": True})[0], 200)
        code, out = srv.handle_api("/api/phone", {"motion": "still", "rms": 0.05,
                                                  "lat": 37.7712, "lon": -122.4413, "acc": 8})
        self.assertEqual((code, out["location"]), (200, True))
        code, out = srv.handle_api("/api/health", {"heart_rate": 72, "steps": 900})
        self.assertEqual((code, out["heart_rate"]), (200, 1))
        comp = cs.read_companion()
        self.assertEqual(comp["checkins"][-1]["kind"], "help")
        self.assertTrue(comp["confirmations"]["4"]["froze"])
        self.assertEqual(comp["watch"]["heart_rate"]["bpm"], 72)
        self.assertEqual(location_io.read_json(self.loc, {})["fix"]["source"], "companion")

    def test_android_features_and_location(self):
        code, out = srv.handle_api("/api/android", {"motion": "walking", "cadence": 98, "lat": 37.7712,
                                                    "lon": -122.4413, "acc_m": 9, "sensors": ["gyroscope"]})
        self.assertEqual((code, out["location"]), (200, True))
        comp = cs.read_companion()
        self.assertEqual(comp["android"]["cadence"], 98)
        self.assertNotIn("lat", comp["android"])
        self.assertEqual(location_io.read_json(self.loc, {})["fix"]["source"], "android")
        self.assertEqual(srv.handle_api("/api/android", {"motion": "x"})[0], 400)

    def test_bad_input_is_400(self):
        self.assertEqual(srv.handle_api("/api/checkin", {"kind": "diagnose"})[0], 400)
        self.assertEqual(srv.handle_api("/api/confirm", {})[0], 400)
        self.assertEqual(srv.handle_api("/api/phone", {"motion": "flying"})[0], 400)
        self.assertEqual(srv.handle_api("/api/checkin", ["x"])[0], 400)
        self.assertEqual(srv.handle_api("/api/nope", {})[0], 404)


class Http(unittest.TestCase):
    """Real HTTPS round trip with the self-signed cert, key checks and static files."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.patches = [mock.patch.object(cs, "COMPANION_FILE", os.path.join(cls.tmp.name, "c.json"))]
        for p in cls.patches:
            p.start()
        cert, key = srv.ensure_cert("127.0.0.1", os.path.join(cls.tmp.name, "certs"))
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(KEY, serve_page=True))
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        cls.httpd.socket = ctx.wrap_socket(cls.httpd.socket, server_side=True)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"https://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.client = ssl.create_default_context(cafile=cert)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        for p in cls.patches:
            p.stop()
        cls.tmp.cleanup()

    def req(self, path, body=None, key=KEY):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["X-StepCue-Key"] = key
        r = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                   headers=headers)
        try:
            with urllib.request.urlopen(r, context=self.client, timeout=5) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def test_page_served_and_cert_valid(self):
        code, data = self.req("/", key=None)
        self.assertEqual(code, 200)
        self.assertIn(b"I need help", data)
        self.assertEqual(self.req("/../companion_server.py", key=None)[0], 404)

    def test_key_required(self):
        self.assertEqual(self.req("/api/status", key=None)[0], 401)
        self.assertEqual(self.req("/api/status", key="wrong")[0], 401)
        code, data = self.req("/api/checkin", {"kind": "ok"})
        self.assertEqual(code, 200)
        code, data = self.req("/api/status")
        self.assertEqual(json.loads(data)["last_checkin"]["kind"], "ok")

    def test_setup_page_and_setup_info(self):
        code, data = self.req("/setup.html", key=None)
        self.assertEqual(code, 200)
        self.assertIn(b"Do you wear an Apple Watch?", data)
        with mock.patch.dict(os.environ, {"STEPCUE_SHORTCUT_URL": "https://www.icloud.com/shortcuts/abc"}):
            code, data = self.req("/api/status")
        info = json.loads(data)["setup"]
        self.assertEqual(info["shortcut_url"], "https://www.icloud.com/shortcuts/abc")
        self.assertTrue(info["watch_url"].endswith(f"/api/health?key={KEY}") or "key=" in info["watch_url"])

    def test_shortcut_body_is_accepted(self):
        code, _ = self.req("/api/health", {"heart_rate": 74, "steps": 3120})
        self.assertEqual(code, 200)
        code, data = self.req("/api/status")
        self.assertEqual(json.loads(data)["watch"]["bpm"], 74)

    def test_android_install_page_and_apk_public(self):
        code, data = self.req("/android", key=None)
        self.assertEqual(code, 200)
        self.assertIn(b"Open StepCue and connect", data)
        if os.path.isfile(os.path.join(srv.STATIC_DIR, "stepcue.apk")):
            with urllib.request.urlopen(self.base + "/stepcue.apk", context=self.client, timeout=5) as r:
                self.assertEqual(r.headers["Content-Type"], "application/vnd.android.package-archive")
                self.assertEqual(r.read(2), b"PK")
        self.assertEqual(self.req("/api/android", {"motion": "still"}, key=None)[0], 401)

    def test_key_in_query_for_watch_apps(self):
        code, _ = self.req(f"/api/health?key={KEY}", {"heartRate": 80}, key=None)
        self.assertEqual(code, 200)


if __name__ == "__main__":
    unittest.main()

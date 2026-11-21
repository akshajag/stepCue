import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auth  # noqa: E402

RFC_SECRET = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"   # base32 of "12345678901234567890" (RFC 6238)


class Totp(unittest.TestCase):
    def test_rfc6238_vectors(self):
        self.assertEqual(auth.totp(RFC_SECRET, t=59), "287082")
        self.assertEqual(auth.totp(RFC_SECRET, t=1111111109), "081804")
        self.assertEqual(auth.totp(RFC_SECRET, t=1234567890), "005924")

    def test_window_allows_one_step_of_drift(self):
        code = auth.totp(RFC_SECRET, t=1000)
        self.assertIsNotNone(auth.match_totp(RFC_SECRET, code, t=1000 + 30))
        self.assertIsNone(auth.match_totp(RFC_SECRET, code, t=1000 + 95))
        self.assertIsNone(auth.match_totp(RFC_SECRET, "12345", t=1000))
        self.assertIsNone(auth.match_totp(RFC_SECRET, "abcdef", t=1000))

    def test_provisioning_uri_and_qr(self):
        uri = auth.provisioning_uri("ABCDEF", "kim@example.com")
        self.assertTrue(uri.startswith("otpauth://totp/StepCue%3Akim%40example.com?"))
        self.assertIn("secret=ABCDEF", uri)
        self.assertTrue(auth.qr_svg(uri).startswith("<svg"))


class Accounts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "users.json")
        data = auth.load_users(self.path)
        auth.add_user(data, " Dr.Lee@Clinic.org ", "Dr Lee", "care", password="correct horse 1")
        auth.add_user(data, "sam@example.com", "Sam", "family")
        auth.save_users(data, self.path)

    def test_password_hash_roundtrip(self):
        h = auth.hash_password("pw-1234567", rounds=1000)
        self.assertTrue(auth.verify_password("pw-1234567", h))
        self.assertFalse(auth.verify_password("pw-7654321", h))
        self.assertFalse(auth.verify_password("x", "garbage"))

    def test_emails_are_normalised_and_unique(self):
        data = auth.load_users(self.path)
        self.assertEqual(auth.find_user(data, "DR.LEE@clinic.org")["role"], "care")
        with self.assertRaises(ValueError):
            auth.add_user(data, "dr.lee@clinic.org", "x", "care")
        with self.assertRaises(ValueError):
            auth.add_user(data, "new@x.org", "x", "admin")
        with self.assertRaises(ValueError):
            auth.add_user(data, "new@x.org", "x", "care", password="short")

    def test_patient_role(self):
        data = auth.load_users(self.path)
        auth.add_user(data, "pat@example.com", "Pat", "patient", password="long enough pw")
        self.assertEqual(auth.find_user(data, "pat@example.com")["role"], "patient")
        self.assertIn("patient", auth.ROLES)

    def test_password_step(self):
        u = auth.check_password("dr.lee@clinic.org", "correct horse 1", self.path)
        self.assertEqual(u["name"], "Dr Lee")
        with self.assertRaises(auth.AuthError):
            auth.check_password("dr.lee@clinic.org", "wrong", self.path)
        with self.assertRaises(auth.AuthError):
            auth.check_password("sam@example.com", "anything", self.path)      # Google-only account

    def test_lockout_after_five_failures(self):
        for _ in range(auth.MAX_FAILS):
            with self.assertRaises(auth.AuthError):
                auth.check_password("dr.lee@clinic.org", "wrong", self.path, now=1000)
        with self.assertRaises(auth.AuthError) as cm:
            auth.check_password("dr.lee@clinic.org", "correct horse 1", self.path, now=1010)
        self.assertIn("Too many attempts", str(cm.exception))
        self.assertTrue(auth.check_password("dr.lee@clinic.org", "correct horse 1", self.path,
                                            now=1000 + auth.LOCK_SEC + 1))

    def test_enrol_then_sign_in_and_no_code_reuse(self):
        sec = auth.new_totp_secret()
        t = 1_800_000_000
        auth.check_code("sam@example.com", auth.totp(sec, t=t), self.path, now=t, secret=sec)
        self.assertEqual(auth.find_user(auth.load_users(self.path), "sam@example.com")["totp_secret"], sec)
        with self.assertRaises(auth.AuthError):                          # same code again
            auth.check_code("sam@example.com", auth.totp(sec, t=t), self.path, now=t + 5)
        u = auth.check_code("sam@example.com", auth.totp(sec, t=t + 30), self.path, now=t + 31)
        self.assertEqual(u["role"], "family")

    def test_code_needs_enrolment(self):
        with self.assertRaises(auth.AuthError):
            auth.check_code("sam@example.com", "123456", self.path)

    def test_reset_and_remove(self):
        sec = auth.new_totp_secret()
        auth.check_code("sam@example.com", auth.totp(sec, t=5000), self.path, now=5000, secret=sec)
        self.assertTrue(auth.reset_2fa("sam@example.com", self.path))
        self.assertNotIn("totp_secret", auth.find_user(auth.load_users(self.path), "sam@example.com"))
        data = auth.load_users(self.path)
        auth.remove_user(data, "sam@example.com")
        self.assertIsNone(auth.find_user(data, "sam@example.com"))


if __name__ == "__main__":
    unittest.main()

import os
import socket
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import esp_bt_bridge as bb  # noqa: E402


class Bridge(unittest.TestCase):
    """The Bluetooth byte stream becomes the same UDP lines and cues the Wi-Fi ESP32 would send."""

    def setUp(self):
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)    # stands in for gait_classifier
        self.rx.bind(("127.0.0.1", 0))
        self.rx.settimeout(2)
        self.addCleanup(self.rx.close)
        self.fwd = bb.Forwarder(target=self.rx.getsockname(), cue_port=0)
        self.addCleanup(self.fwd.cue.close)
        self.addCleanup(self.fwd.out.close)

    def test_lines_split_across_chunks_and_garbage_dropped(self):
        n = self.fwd.feed(b"MPU6050 ready\r\n1200,0.012,-0.981,0.05,1.2")
        self.assertEqual(n, 0)
        n = self.fwd.feed(b"0,-3.5,0.4\n1220,0.01,-0.98,0.05,1.1,-3.4,0.5\nbad,line\n")
        self.assertEqual(n, 2)
        self.assertEqual(self.fwd.bad, 2)
        data, addr = self.rx.recvfrom(256)
        self.assertEqual(data, b"1200,0.012,-0.981,0.05,1.20,-3.5,0.4\n")
        self.assertEqual(addr[1], self.fwd.cue.getsockname()[1])      # sent from the cue socket

    def test_cues_come_back_through_the_bridge(self):
        self.fwd.feed(b"1,0,0,1,0,0,0\n")
        _, addr = self.rx.recvfrom(256)
        self.rx.sendto(b"CUE:100:1", addr)                             # what gait_classifier sends back
        self.assertEqual(self.fwd.next_cue(), "CUE:100:1")
        self.rx.sendto(b"rm -rf", addr)
        self.assertIsNone(self.fwd.next_cue())                         # only real cue commands pass
        self.rx.sendto(b"STOP", addr)
        self.assertEqual(self.fwd.next_cue(), "STOP")


if __name__ == "__main__":
    unittest.main()

"""Local disconnect evidence is not evidence of server resource release."""
import contextlib
import http.server
import json
import socket
import threading
import unittest
from unittest.mock import patch

from w34perf.transport import stream


def event(delta, finish=None):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


@contextlib.contextmanager
def server(events, *, wait=False, status=200):
    disconnected = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for item in events:
                    raw = item if isinstance(item, str) else json.dumps(item)
                    self.wfile.write(("data: " + raw + "\n\n").encode())
                    self.wfile.flush()
                if wait:
                    self.connection.settimeout(2)
                    if self.connection.recv(1) == b"":
                        disconnected.set()
            except (BrokenPipeError, ConnectionResetError):
                disconnected.set()
            except socket.timeout:
                pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield {"base_url": f"http://127.0.0.1:{httpd.server_port}/v1"}, disconnected
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


class DisconnectTests(unittest.TestCase):
    def test_threshold_counts_output_events_not_fields_or_metadata(self):
        frames = [event({"role": "assistant"}), {"usage": {"total_tokens": 1}},
                  event({"content": "", "reasoning": ""}),
                  event({"content": "a", "reasoning_content": "b"}),
                  event({"tool_calls": [{"index": 0, "function": {"name": "f"}}]})]
        with server(frames, wait=True) as (engine, closed):
            result = stream(engine, {}, 2, cancel_after_output_events=2)
            self.assertEqual(result["status"], "cancelled")
            self.assertIsNone(result["error"])
            self.assertTrue(result["cancellation_requested"])
            self.assertTrue(result["cancellation_observed"])
            self.assertEqual(len(result["events"]), 5)
            self.assertTrue(closed.wait(1), "response socket did not close")

    def test_disconnect_racing_closed_peer_is_still_local_injection(self):
        # A peer can close after buffering output but before the client's
        # injection. Local close still succeeds; no server acknowledgment is claimed.
        with server([event({"content": "buffered"})]) as (engine, _):
            with patch("socket.socket.shutdown", side_effect=OSError("socket not connected")):
                result = stream(engine, {}, 2, cancel_after_output_events=1)
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(result["cancellation_observed"])
        self.assertIsNone(result["error"])

    def test_terminal_wins_even_on_threshold_event(self):
        for threshold in (1, 3):
            with self.subTest(threshold=threshold):
                with server([event({"content": "done"}, "stop"), "[DONE]"]) as (engine, _):
                    result = stream(engine, {}, 2, cancel_after_output_events=threshold)
                self.assertEqual(result["status"], "ok")
                self.assertTrue(result["cancellation_requested"])
                self.assertFalse(result["cancellation_observed"])

    def test_invalid_before_network(self):
        for value in (True, False, 0, -1, 1.0, "1", [], {}):
            with self.subTest(value=value), patch("http.client.HTTPConnection.connect") as connect:
                with self.assertRaises(ValueError):
                    stream({}, {}, 1, cancel_after_output_events=value)
                connect.assert_not_called()

    def test_errors_and_timeout_do_not_count_as_injection(self):
        cases = [([{"error": "failed"}], False, 200, "error"),
                 ([event({"content": "a"})], False, 200, "error"),
                 ([], True, 503, "error"),
                 ([event({"content": "a"})], True, 200, "timeout")]
        for frames, wait, status, expected in cases:
            with self.subTest(expected=expected, status=status):
                with server(frames, wait=wait, status=status) as (engine, closed):
                    result = stream(engine, {}, .1, cancel_after_output_events=2)
                    if wait:
                        self.assertTrue(closed.wait(1))
                self.assertEqual(result["status"], expected)
                self.assertIsNotNone(result["error"])
                self.assertTrue(result["cancellation_requested"])
                self.assertFalse(result["cancellation_observed"])

    def test_default_unchanged(self):
        with server([event({"content": "ok"}, "stop"), "[DONE]"]) as (engine, _):
            result = stream(engine, {}, 2)
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("cancellation_requested", result)


if __name__ == "__main__":
    unittest.main()

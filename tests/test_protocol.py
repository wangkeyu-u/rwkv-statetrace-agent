from __future__ import annotations

import unittest

from statetrace.models import ErrorCode, FinalAction, ToolCall
from statetrace.protocol import ProtocolError, parse_action


class ProtocolTests(unittest.TestCase):
    def test_parses_fenced_tool_action(self):
        action = parse_action(
            '```json\n{"type":"tool_call","thought_summary":"inspect",'
            '"tool":"list_files","arguments":{"path":"."}}\n```'
        )
        self.assertIsInstance(action, ToolCall)
        self.assertEqual(action.tool, "list_files")

    def test_parses_final_report_to_json_lists(self):
        action = parse_action(
            '{"type":"final","tool":"finish_report","arguments":{'
            '"summary":"Found it","findings":[{"file":"a.py","line":1,'
            '"claim":"Bug","evidence_ids":["obs-001"]}],'
            '"verification":{},"recommendations":["Fix it"]}}'
        )
        self.assertIsInstance(action, FinalAction)
        self.assertEqual(action.report.as_dict()["findings"][0]["evidence_ids"], ["obs-001"])

    def test_invalid_json_becomes_safe_feedback(self):
        with self.assertRaises(ProtocolError) as caught:
            parse_action("```json\n{broken\n```")
        self.assertEqual(caught.exception.code, ErrorCode.INVALID_JSON)
        self.assertEqual(caught.exception.as_observation().status, "error")

    def test_rejects_wrong_argument_shape_and_finish_type(self):
        with self.assertRaises(ProtocolError):
            parse_action('{"type":"tool_call","tool":"read_file","arguments":[]}')
        with self.assertRaises(ProtocolError):
            parse_action('{"type":"tool_call","tool":"finish_report","arguments":{}}')


if __name__ == "__main__":
    unittest.main()

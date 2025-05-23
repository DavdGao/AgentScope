# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Unit test for serialization."""
import json
import unittest

from agentscope.message import Msg
from agentscope.serialize import serialize, deserialize


class SerializationTest(unittest.TestCase):
    """The test cases for serialization."""

    def test_serialize(self) -> None:
        """Test the serialization function."""

        msg1 = Msg("A", "A", "assistant")
        msg2 = Msg("B", "B", "assistant")

        serialized_msg1 = serialize(msg1)
        deserialized_msg1 = deserialize(serialized_msg1)
        self.assertTrue(isinstance(serialized_msg1, str))
        self.assertTrue(isinstance(deserialized_msg1, Msg))

        msg1_dict = json.loads(serialized_msg1)
        self.assertDictEqual(
            msg1_dict,
            {
                "id": msg1.id,
                "name": "A",
                "content": "A",
                "role": "assistant",
                "timestamp": msg1.timestamp,
                "metadata": None,
                "__module__": "agentscope.message.msg",
                "__name__": "Msg",
            },
        )

        serialized_list = serialize([msg1, msg2])
        deserialized_list = deserialize(serialized_list)
        self.assertTrue(isinstance(serialized_list, str))
        self.assertTrue(
            isinstance(deserialized_list, list)
            and len(deserialized_list) == 2
            and all(isinstance(msg, Msg) for msg in deserialized_list),
        )

        dict_list = json.loads(serialized_list)
        self.assertListEqual(
            dict_list,
            [
                {
                    "id": msg1.id,
                    "name": "A",
                    "content": "A",
                    "role": "assistant",
                    "timestamp": msg1.timestamp,
                    "metadata": None,
                    "__module__": "agentscope.message.msg",
                    "__name__": "Msg",
                },
                {
                    "id": msg2.id,
                    "name": "B",
                    "content": "B",
                    "role": "assistant",
                    "timestamp": msg2.timestamp,
                    "metadata": None,
                    "__module__": "agentscope.message.msg",
                    "__name__": "Msg",
                },
            ],
        )

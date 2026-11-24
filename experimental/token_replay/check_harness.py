import importlib.metadata
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

from aiperf.common.enums import ConversationContextMode
from aiperf.config.flags import CLIConfig
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader

from infx.bench.token_replay import export_agentx


def check_harness() -> dict:
    sessions = {"session-a": [
        {"session_id": "session-a", "input_ids": [1, 2], "output_ids": [8, 9], "delay_seconds": 0},
        {"session_id": "session-a", "input_ids": [1, 2, 8, 9, 3], "output_ids": [7], "delay_seconds": 0.75},
    ]}
    generator = Mock()
    generator._decoded_cache = {}
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "replay.jsonl"
        export_agentx(sessions, path, "served-model", 42)
        loader = MooncakeTraceDatasetLoader(
            filename=path, cfg=CLIConfig(model_names=["served-model"], url="http://localhost:8000"),
            prompt_generator=generator,
        )
        conversations = loader.convert_to_conversations(loader.load_dataset())
    assert len(conversations) == 1
    conversation = conversations[0]
    assert conversation.context_mode == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
    assert [turn.delay for turn in conversation.turns] == [0, 750]
    assert [turn.raw_payload["prompt"] for turn in conversation.turns] == [[1, 2], [1, 2, 8, 9, 3]]
    assert [turn.raw_payload["replay_token_ids"] for turn in conversation.turns] == [[8, 9], [7]]
    generator.generate.assert_not_called()
    generator._build_token_sequence.assert_not_called()
    return {"agentx_harness": importlib.metadata.version("agentx-harness"), "real_loader": "passed", "turns": 2}


if __name__ == "__main__":
    print(json.dumps(check_harness(), indent=2))

import argparse
import json

from transformers import AutoTokenizer

from prepare_exgentic import encode_span


def check_target_template(model: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    arguments = {"command": "printf 'hello'", "timeout": 5}
    call = {"role": "assistant", "parts": [
        {"type": "tool_call", "name": "run", "id": "c1", "arguments": arguments}
    ]}
    user = {"role": "user", "parts": [{"type": "text", "content": "Run the command."}]}
    span = {
        "start_time": "2026-01-01T00:00:00+00:00",
        "end_time": "2026-01-01T00:00:01+00:00",
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.input.messages": json.dumps([user]),
            "gen_ai.output.messages": json.dumps([call]),
        },
    }
    encoded = encode_span(span, tokenizer)
    output = tokenizer.decode(encoded["output_ids"], skip_special_tokens=False)
    parsed = json.loads(output.split("<tool_call>\n", 1)[1].split("\n</tool_call>", 1)[0])
    assert parsed == {"name": "run", "arguments": arguments}, parsed
    span["attributes"]["gen_ai.input.messages"] = json.dumps([user, call, user])
    encoded = encode_span(span, tokenizer)
    prompt = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    historical = json.loads(prompt.split("<tool_call>\n", 1)[1].split("\n</tool_call>", 1)[0])
    assert historical == parsed, historical
    call["parts"][0]["arguments"] = "not-json"
    span["attributes"]["gen_ai.output.messages"] = json.dumps([call])
    try:
        encode_span(span, tokenizer)
    except ValueError as error:
        assert str(error) == "tool_arguments_are_not_valid_json", error
    else:
        raise AssertionError("malformed tool arguments were accepted")
    return {"target_model": model, "tool_argument_object_preserved": True,
            "historical_tool_arguments_preserved": True, "malformed_arguments_rejected": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    print(json.dumps(check_target_template(args.model), indent=2))

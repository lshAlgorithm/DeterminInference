import argparse
import json
import urllib.request


def check_cache_salt(url: str) -> dict:
    rows = []
    for salt in ("isolation-a", "isolation-b", "isolation-a"):
        payload = {
            "input_ids": list(range(500, 564)),
            "extra_key": salt,
            "sampling_params": {
                "max_new_tokens": 2,
                "ignore_eos": True,
                "temperature": 0,
                "replay_token_ids": [7, 8],
            },
        }
        request = urllib.request.Request(
            url.rstrip("/") + "/generate",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        assert result["output_ids"] == [7, 8], result
        rows.append(
            {"salt": salt, "cached_tokens": result["meta_info"]["cached_tokens"]}
        )
    assert [row["cached_tokens"] for row in rows[:2]] == [0, 0], rows
    assert rows[2]["cached_tokens"] >= 63, rows
    return {
        "passed": True,
        "seed": 42,
        "rows": rows,
        "scope": "identical token IDs isolated across salts and reused within a salt",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    print(json.dumps(check_cache_salt(args.url), indent=2))

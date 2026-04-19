"""Test utility functions."""

import sys
sys.path.insert(0, 'src')

from utils import count_tokens, format_chunk

def test_count_tokens():
    text = "hello world test"
    assert count_tokens(text) == 3
    print("✓ test_count_tokens passed")

def test_format_chunk():
    text = "Sample text"
    source = "test.pdf"
    result = format_chunk(text, source)
    assert "[test.pdf]" in result
    print("✓ test_format_chunk passed")

if __name__ == "__main__":
    test_count_tokens()
    test_format_chunk()
    print("\n✓ All tests passed!")
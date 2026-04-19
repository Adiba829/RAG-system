"""Utility functions."""

def count_tokens(text: str) -> int:
    """Count tokens in text (rough estimate)."""
    return len(text.split())

def format_chunk(text: str, source: str) -> str:
    """Format a chunk for display."""
    return f"[{source}]\n{text}"

if __name__ == "__main__":
    sample = "This is a sample text with ten words in it total here"
    print(f"Token count: {count_tokens(sample)}")
    print(f"Formatted: {format_chunk(sample, 'test.pdf')}")
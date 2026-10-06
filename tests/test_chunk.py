from ingest.chunk import chunk_python

SRC = '''import os

X = 1

class A(Base):
    """Doc."""
    def f(self):
        return 1

@deco
def g():
    return 2
'''

def test_chunk_python():
    chunks = chunk_python("a.py", SRC)
    symbols = {c["symbol"] for c in chunks}
    assert {"<module>", "A", "A.f", "g"} <= symbols
    g = next(c for c in chunks if c["symbol"] == "g")
    assert g["start_line"] == 10  # decorator line is included
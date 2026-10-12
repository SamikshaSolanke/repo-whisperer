from query.answer import cited_indices


def test_cited_indices_handles_all_bracket_styles():
    text = "a [1] b 【3†L5-L7】 c [2][4] d 【2】 e [9]"
    assert cited_indices(text, 4) == [1, 2, 3, 4]      # [9] is out of range and ignored
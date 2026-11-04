"""Regression tests for the multimodal guard in ``vipera.optimize.shaping._shrink``.

A message whose ``content`` is a list of parts must never be rewritten: the
shaping layer cannot turn a cut repr of that list back into the model. The
guard in ``_shrink`` leaves such a message untouched and reports ``False``.
"""

from vipera.optimize.shaping import _shrink


def _multimodal_message() -> dict:
    """Build a multimodal message whose content is a list of parts."""
    content = [{"type": "text", "text": "x" * 20000}]
    return {"role": "user", "content": content}


def test_shrink_multimodal_content_is_left_untouched() -> None:
    msg = _multimodal_message()
    original = msg["content"]

    assert _shrink(msg, 100) is False
    assert msg["content"] is original
    assert not isinstance(msg["content"], str)

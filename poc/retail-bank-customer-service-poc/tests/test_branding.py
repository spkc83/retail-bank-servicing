from branding import GRADIO_CSS


def test_gradio_shell_layout_and_assistant_table_css_contract() -> None:
    assert "max-width: 1680px" in GRADIO_CSS
    assert "margin: 0 auto" in GRADIO_CSS
    assert ".harbor-layout" in GRADIO_CSS
    assert "@media (max-width: 900px)" in GRADIO_CSS
    assert "flex-direction: column" in GRADIO_CSS
    assert ".harbor-chat .message.bot table" in GRADIO_CSS
    assert "overflow-x: auto" in GRADIO_CSS
    assert "white-space: nowrap" in GRADIO_CSS
    assert "word-break: normal" in GRADIO_CSS

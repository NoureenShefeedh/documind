import httpx
import pytest

from app.rag.llm import (
    NOT_FOUND,
    AnthropicAnswerer,
    ExtractiveAnswerer,
    LLMError,
    Source,
    build_prompt,
)

SOURCES = [
    Source("handbook.txt", 0, "Employees get 24 days of paid leave. The office opens at 9am.", 0.9),
    Source("faq.md", 2, "Laptops are replaced every three years.", 0.4),
]


def test_prompt_contains_numbered_sources_and_question():
    system, user = build_prompt("How many leave days?", SOURCES)
    assert '<source id="1" file="handbook.txt">' in user
    assert '<source id="2" file="faq.md">' in user
    assert "How many leave days?" in user
    assert "ONLY" in system


def test_prompt_strips_quotes_from_filenames():
    _, user = build_prompt("q", [Source('a"b.txt', 0, "text", 1.0)])
    assert 'file="ab.txt"' in user


def test_extractive_picks_relevant_sentence_with_citation():
    answer = ExtractiveAnswerer().answer("How many days of paid leave?", SOURCES)
    assert "24 days of paid leave" in answer
    assert "[1]" in answer


def test_extractive_says_not_found():
    assert ExtractiveAnswerer().answer("What is the CEO's favourite colour?", SOURCES) == NOT_FOUND


def test_extractive_dedupes_overlapping_chunks():
    dup = [Source("a.txt", 0, "Cats sleep a lot.", 0.9), Source("a.txt", 1, "Cats sleep a lot.", 0.8)]
    assert ExtractiveAnswerer().answer("do cats sleep", dup).count("Cats sleep a lot.") == 1


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_anthropic_answerer_parses_text_blocks():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"content": [{"type": "text", "text": "24 days [1]."}]})

    answerer = AnthropicAnswerer("sk-test", "some-model", client=_client(handler))
    assert answerer.answer("How many days?", SOURCES) == "24 days [1]."
    assert seen["headers"]["x-api-key"] == "sk-test"
    assert "How many days?" in seen["body"]


def test_anthropic_answerer_wraps_http_errors():
    answerer = AnthropicAnswerer("k", "m", client=_client(lambda r: httpx.Response(500)))
    with pytest.raises(LLMError):
        answerer.answer("q", SOURCES)


def test_anthropic_answerer_rejects_empty_answer():
    answerer = AnthropicAnswerer(
        "k", "m", client=_client(lambda r: httpx.Response(200, json={"content": []}))
    )
    with pytest.raises(LLMError):
        answerer.answer("q", SOURCES)

"""Tests for grounded answer synthesis: citations + abstention."""

from application.synthesis import SynthesizeAnswerUseCase, strip_llm_output
from domain.models import SearchHit
from tests.fakes import FakeLLM


def _hit(cid, doc_id="d1", text="the deployment port is 8080"):
    return SearchHit(
        chunk_id=cid,
        doc_id=doc_id,
        source_name="deploy.txt",
        text=text,
        score=0.5,
        page_number=3,
        section_heading="Ops",
        chunk_index=0,
    )


# ----------------------------------------------------------------------
# strip_llm_output
# ----------------------------------------------------------------------
def test_strip_think_tags():
    assert strip_llm_output("   <think>hidden reasoning</think>  Yes [1]") == "Yes [1]"


def test_strip_fences():
    assert strip_llm_output("```\ncode\n```") == "code"
    assert strip_llm_output("```json\n{\"a\": 1}\n```") == '{"a": 1}'


def test_strip_none_and_empty():
    assert strip_llm_output(None) == ""
    assert strip_llm_output("") == ""
    assert strip_llm_output("   ") == ""


# ----------------------------------------------------------------------
# SynthesizeAnswerUseCase
# ----------------------------------------------------------------------
def test_citations_mapped_to_chunks():
    llm = FakeLLM("The port is 8080 because the config says so. See [1].")
    synth = SynthesizeAnswerUseCase(llm_provider=llm)
    resp = synth.execute("what port?", [_hit("c1", text="the deployment port is 8080")])
    assert resp.abstained is False
    assert len(resp.citations) == 1
    cit = resp.citations[0]
    assert cit.index == 1
    assert cit.chunk_id == "c1"
    assert cit.source_name == "deploy.txt"
    assert cit.page_number == 3
    assert cit.section_heading == "Ops"
    assert "8080" in cit.snippet


def test_multiple_citations():
    llm = FakeLLM("X and Y. [1][2]")
    synth = SynthesizeAnswerUseCase(llm_provider=llm)
    hits = [_hit("c1", text="aaa"), _hit("c2", text="bbb")]
    resp = synth.execute("q", hits)
    assert [c.index for c in resp.citations] == [1, 2]


def test_out_of_range_citation_ignored():
    llm = FakeLLM("No citations here [99].")
    synth = SynthesizeAnswerUseCase(llm_provider=llm)
    resp = synth.execute("q", [_hit("c1")])
    assert resp.citations == []
    assert resp.abstained is False


def test_abstain_marker():
    llm = FakeLLM("ABSTAIN")
    synth = SynthesizeAnswerUseCase(llm_provider=llm)
    resp = synth.execute("secret question?", [_hit("c1")])
    assert resp.abstained is True
    assert "not found" in resp.answer.lower() or "could not" in resp.answer.lower()
    assert resp.citations == []


def test_empty_llm_response_abstains():
    llm = FakeLLM("")
    synth = SynthesizeAnswerUseCase(llm_provider=llm)
    resp = synth.execute("q", [_hit("c1")])
    assert resp.abstained is True


def test_no_hits_no_llm_call():
    llm = FakeLLM("SHOULD NOT BE CALLED")
    synth = SynthesizeAnswerUseCase(llm_provider=llm)
    resp = synth.execute("q", [])
    assert resp.abstained is True
    assert len(llm.prompts) == 0


def test_model_recorded():
    llm = FakeLLM("Answer [1].")
    synth = SynthesizeAnswerUseCase(llm_provider=llm)
    resp = synth.execute("q", [_hit("c1")])
    assert resp.model == "fake-model"
    assert resp.hits_count == 1
    assert resp.latency_ms >= 0

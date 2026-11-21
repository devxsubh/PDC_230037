from decimal import Decimal

from agent_fixtures import record
from finevidence.agent.evidence import (
    clean_cell,
    conflicting_mentions,
    extract_mentions,
    label_match,
    literal_for,
    table_mentions,
    text_mentions,
)
from finevidence.retrieval import EvidenceSegment
from finevidence.retrieval.lexical import (
    Bm25Retriever,
    RankingFileRetriever,
    record_segments,
    tokenize,
)

HEADER = ("( in millions )", "2017", "2016")


def _row(key, *cells):
    return EvidenceSegment(key, "table", " | ".join(cells), tuple(cells), HEADER)


def test_tokenize_drops_stopwords_and_number_commas():
    assert tokenize("What was the Revenue of 1,100 in 2017?") == ["revenue", "1100", "2017"]


def test_bm25_ranks_the_matching_row_first():
    ranked = Bm25Retriever().rank(record(), "net sales 2017")
    assert ranked[0].key == "table_1"
    assert [item.rank for item in ranked] == list(range(1, len(ranked) + 1))
    assert len(ranked) == len(record_segments(record()))


def test_ranking_file_retriever_replays_saved_order_and_appends_the_rest():
    ranked = RankingFileRetriever({record()["id"]: ["text_2", "table_3"]}).rank(record(), "x")
    assert [item.key for item in ranked[:2]] == ["text_2", "table_3"]
    assert {item.key for item in ranked} == {s.key for s in record_segments(record())}


def test_table_mentions_take_row_label_column_and_period():
    mentions = table_mentions(_row("table_1", "net sales", "$ 1100", "$ 1000"))
    assert [(m.literal, m.periods, m.label) for m in mentions] == [
        ("1100", ("2017",), "net sales"),
        ("1000", ("2016",), "net sales"),
    ]


def test_accounting_parentheses_become_negative_literals():
    mentions = table_mentions(_row("table_3", "operating income", "( 50 )", "120"))
    assert mentions[0].value == Decimal("-50")
    assert mentions[0].literal == "-50"


def test_echoed_values_are_cleaned_before_parsing():
    assert clean_cell("$ -4063 ( 4063 )") == "$ -4063"
    assert clean_cell("32.26% ( 32.26 % )") == "32.26%"
    assert clean_cell("( 50 )") == "( 50 )"
    mentions = table_mentions(_row("table_2", "total", "$ -4063 ( 4063 )", "$ 3698"))
    assert [m.literal for m in mentions] == ["-4063", "3698"]


def test_header_row_with_data_yields_mentions_without_column():
    segment = EvidenceSegment(
        "table_0", "table", "", ("balance at may 31 2014", "$ -7199"), ("balance", "$ -7199")
    )
    (mention,) = table_mentions(segment)
    assert (mention.literal, mention.periods, mention.column) == ("-7199", ("2014",), "")


def test_text_mentions_skip_years_and_take_the_nearest_year():
    segment = EvidenceSegment("text_2", "text", "interest expense was $ 40 million in 2017 .")
    (mention,) = text_mentions(segment)
    assert (mention.literal, mention.periods) == ("40", ("2017",))


def test_label_match_prefers_the_row_label():
    row = table_mentions(_row("table_1", "net sales", "$ 1100", "$ 1000"))[0]
    other = table_mentions(_row("table_2", "cost of sales", "600", "550"))[0]
    assert label_match(("net", "sales"), row) > label_match(("net", "sales"), other)
    assert label_match((), row) == 0.0


def test_conflicting_values_for_the_same_label_and_period_are_grouped():
    mentions = extract_mentions(
        [
            _row("table_1", "net sales", "$ 1100", "$ 1000"),
            _row("table_5", "net sales", "$ 1150", "$ 1000"),
        ]
    )
    (group,) = conflicting_mentions(mentions)
    assert {m.literal for m in group} == {"1100", "1150"}


def test_literal_rendering_has_no_exponent():
    assert literal_for(Decimal("1.20E+3")) == "1200"
    assert literal_for(Decimal("-0")) == "0"

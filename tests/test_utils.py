"""Regression tests using synthetic manifests; no speech corpus is required."""

from copy import deepcopy

import pytest
from lhotse import CutSet, MonoCut
from lhotse.supervision import AlignmentItem, SupervisionSegment

from fastmss.utils import split_monocut_at_pauses, split_monocuts_batch


def make_cut(items, start=0.0, duration=5.0, supervision_start=0.0, identifier="cut"):
    return MonoCut(
        id=identifier,
        start=start,
        duration=duration,
        channel=0,
        supervisions=[
            SupervisionSegment(
                id=identifier + "-sup",
                recording_id="recording",
                start=supervision_start,
                duration=duration - supervision_start,
                channel=0,
                speaker="speaker",
                text="original text",
                custom={"source": "synthetic"},
                alignment={"word": items},
            )
        ],
        custom={"dataset": "synthetic"},
    )


def test_nested_event_is_not_split_and_trailing_boundary_uses_longest_interval():
    items = [
        AlignmentItem("[event]", 0.2, 3.0),
        AlignmentItem("one", 0.4, 0.2),
        AlignmentItem("two", 2.0, 0.2),
    ]
    result = split_monocut_at_pauses(make_cut(items))
    assert len(result) == 1
    assert result[0].start == pytest.approx(0.1)
    assert (result[0].start + result[0].duration) == pytest.approx(4.1)
    assert [a.symbol for a in result[0].supervisions[0].alignment["word"]] == [
        "[event]",
        "one",
        "two",
    ]


def test_gap_after_overlapping_group_splits_at_union_boundary():
    items = [
        AlignmentItem("event", 0.0, 3.0),
        AlignmentItem("word", 0.2, 0.2),
        AlignmentItem("after", 4.0, 1.0),
    ]
    result = split_monocut_at_pauses(make_cut(items))
    assert [cut.start for cut in result] == pytest.approx([0.0, 3.5])
    assert [cut.start + cut.duration for cut in result] == pytest.approx([3.5, 5.0])
    for cut in result:
        assert cut.custom == {"dataset": "synthetic"}
        assert cut.supervisions[0].custom == {"source": "synthetic"}
        for item in cut.supervisions[0].alignment["word"]:
            assert item.start >= 0
            assert item.start + item.duration <= cut.duration


def test_unsorted_alignments_produce_same_result():
    items = [
        AlignmentItem("event", 0.0, 3.0),
        AlignmentItem("word", 0.2, 0.2),
        AlignmentItem("after", 4.0, 1.0),
    ]
    expected = split_monocut_at_pauses(make_cut(items))
    actual = split_monocut_at_pauses(make_cut(list(reversed(items))))
    assert [cut.to_dict() for cut in actual] == [cut.to_dict() for cut in expected]


@pytest.mark.parametrize("offset", [0.0, 10.0])
def test_no_split_normalizes_cut_or_recording_frame_without_mutating_input(offset):
    cut = make_cut(
        [AlignmentItem("one", offset, 1.0), AlignmentItem("two", offset + 1.0, 1.0)],
        start=10.0,
        duration=2.0,
    )
    original = deepcopy(cut.to_dict())
    result = split_monocut_at_pauses(cut)
    assert len(result) == 1
    assert [item.start for item in result[0].supervisions[0].alignment["word"]] == [
        0.0,
        1.0,
    ]
    assert cut.to_dict() == original


def test_supervision_relative_frame_is_still_supported():
    cut = make_cut(
        [AlignmentItem("one", 0.0, 0.5)],
        start=10.0,
        duration=2.0,
        supervision_start=0.5,
    )
    # Both relative frames fit; preserve upstream's documented cut-first choice.
    result = split_monocut_at_pauses(cut)
    assert result[0].supervisions[0].alignment["word"][0].start == 0.0


def test_inconsistent_alignment_is_rejected():
    with pytest.raises(ValueError, match="do not fit"):
        split_monocut_at_pauses(make_cut([AlignmentItem("outside", 50.0, 1.0)]))


@pytest.mark.parametrize("drop", [True, False])
def test_missing_alignment(drop):
    cut = make_cut([])
    cut.supervisions[0].alignment = None
    assert split_monocut_at_pauses(cut, drop_without_alignment=drop) == (
        [] if drop else [cut]
    )


def test_disabled_splitting_returns_original():
    cut = make_cut([AlignmentItem("word", 0.0, 1.0)])
    assert split_monocut_at_pauses(cut, pause_threshold=None) == [cut]


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="positive integer"):
        split_monocuts_batch([], batch_size=batch_size)


@pytest.mark.parametrize("num_jobs", [1, 2])
def test_batches_preserve_all_cuts_order_and_alignments(num_jobs, tmp_path):
    cuts = [
        make_cut(
            [AlignmentItem("one", 0.0, 1.0), AlignmentItem("two", 2.0, 1.0)],
            duration=3.0,
            identifier=f"cut-{i}",
        )
        for i in range(5)
    ]
    path = tmp_path / "cuts.jsonl.gz"
    CutSet.from_cuts(cuts).to_file(path)
    expected = [part.to_dict() for cut in cuts for part in split_monocut_at_pauses(cut)]
    result = split_monocuts_batch(
        CutSet.from_file(path), num_jobs=num_jobs, batch_size=2
    )
    assert [cut.to_dict() for cut in result] == expected


def test_empty_batch_input():
    assert len(split_monocuts_batch([], batch_size=2)) == 0


def test_large_input_is_consumed_one_batch_at_a_time(monkeypatch):
    import fastmss.utils as utils

    consumed = []
    processed = []

    class Cuts:
        def __len__(self):
            return 5

        def __iter__(self):
            for i in range(5):
                consumed.append(i)
                yield make_cut(
                    [AlignmentItem("word", 0.0, 1.0)], duration=1.0, identifier=str(i)
                )

    def parallel_map(function, batch, num_jobs):
        assert len(batch) <= 2
        assert len(consumed) - len(processed) == len(batch)
        for cut in batch:
            processed.append(cut.id)
            yield function(cut)

    monkeypatch.setattr(utils, "parallel_map", parallel_map)
    assert len(split_monocuts_batch(Cuts(), batch_size=2)) == 5

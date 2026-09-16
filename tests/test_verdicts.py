"""verdicts: what the user thought of a clip, and the context that produced it."""
import json

import pytest

import verdicts as v


PIPELINE_CLIP = {
    "video_title_for_youtube_short": "I let Jake Paul body shot me",
    "viral_hook_text": "I let a pro boxer hit me in the body",
    "start": 726.27, "end": 760.6, "predicted_score": 88,
}
AGENT_CLIP = {"title": "Agent clip", "hook": "Agent hook",
              "start": 1.0, "end": 20.0, "score": 77, "selected_by": "agent"}
CONTEXT = {"mode": "classical", "provider": "gemini",
           "model": "gemini-3.1-flash-lite", "niche": None}


class TestNormalize:
    def test_accepts_the_closed_values(self):
        assert v.normalize("good") == ("good", None)
        assert v.normalize("BAD", "No_Payoff") == ("bad", "no_payoff")

    def test_blank_reason_is_none_not_an_error(self):
        assert v.normalize("bad", "   ") == ("bad", None)

    @pytest.mark.parametrize("verdict,reason", [
        ("great", None), ("", None), (None, None), ("good", "because I said so"),
        ("good", "meh"),
    ])
    def test_rejects_anything_outside_the_lists(self, verdict, reason):
        with pytest.raises(v.VerdictError):
            v.normalize(verdict, reason)

    def test_reasons_are_a_closed_list(self):
        # Free text on a row designed to outlive its job cannot be counted.
        assert "no_payoff" in v.REASONS and "nothing_to_watch" in v.REASONS
        assert all(r.islower() and " " not in r for r in v.REASONS)


class TestBuildRow:
    def test_carries_the_clip_and_how_it_was_made(self):
        row = v.build_row("job1", 0, "good", clip=PIPELINE_CLIP,
                          context=CONTEXT, user="u1", now=100.0)
        assert row["job_id"] == "job1" and row["clip_index"] == 0
        assert row["verdict"] == "good" and row["reason"] is None
        assert row["at"] == 100.0 and row["user"] == "u1"
        assert row["clip"]["title"] == "I let Jake Paul body shot me"
        assert row["clip"]["predicted_score"] == 88
        # Provenance is the whole point: without it the data cannot say
        # whether a later mode helped.
        assert row["context"]["mode"] == "classical"
        assert row["context"]["provider"] == "gemini"

    def test_reads_the_agent_field_names_too(self):
        row = v.build_row("j", 1, "bad", "boring", clip=AGENT_CLIP)
        assert row["clip"]["title"] == "Agent clip"
        assert row["clip"]["hook"] == "Agent hook"
        assert row["clip"]["predicted_score"] == 77
        assert row["clip"]["selected_by"] == "agent"

    def test_an_unknown_context_key_is_kept(self):
        # A field a later phase adds must land on the row without changing this.
        row = v.build_row("j", 0, "good", context={"vision": "frames"})
        assert row["context"]["vision"] == "frames"

    def test_a_missing_clip_does_not_raise(self):
        row = v.build_row("j", 0, "good")
        assert row["clip"]["title"] == "" and row["clip"]["predicted_score"] is None

    def test_an_invalid_verdict_raises(self):
        with pytest.raises(v.VerdictError):
            v.build_row("j", 0, "excellent")


class TestStore:
    def test_append_and_read_round_trip(self, tmp_path):
        d = str(tmp_path)
        v.append_row(d, v.build_row("j", 0, "good", clip=PIPELINE_CLIP, now=1.0))
        v.append_row(d, v.build_row("j", 1, "bad", "boring", now=2.0))
        rows = v.read_rows(d)
        assert [r["clip_index"] for r in rows] == [0, 1]

    def test_no_store_yet_reads_empty(self, tmp_path):
        assert v.read_rows(str(tmp_path)) == []

    def test_a_damaged_line_is_skipped_not_fatal(self, tmp_path):
        d = str(tmp_path)
        v.append_row(d, v.build_row("j", 0, "good", now=1.0))
        with open(v.store_path(d), "a", encoding="utf-8") as fh:
            fh.write("{not json\n\n")
        v.append_row(d, v.build_row("j", 1, "good", now=2.0))
        assert len(v.read_rows(d)) == 2

    def test_re_rating_is_another_append_and_the_latest_wins(self, tmp_path):
        d = str(tmp_path)
        v.append_row(d, v.build_row("j", 0, "good", now=1.0))
        v.append_row(d, v.build_row("j", 0, "bad", "wrong_moment", now=2.0))
        current = v.latest(v.read_rows(d))
        assert len(current) == 1
        assert current[("j", 0)]["verdict"] == "bad"

    def test_for_job_is_keyed_by_clip_index(self, tmp_path):
        d = str(tmp_path)
        v.append_row(d, v.build_row("a", 0, "good", now=1.0))
        v.append_row(d, v.build_row("b", 0, "bad", "boring", now=1.0))
        by_clip = v.for_job(v.read_rows(d), "a")
        assert set(by_clip) == {0}
        assert by_clip[0]["verdict"] == "good"


class TestSummarise:
    def test_counts_and_the_score_split(self):
        rows = [
            v.build_row("j", 0, "good", clip={"predicted_score": 90}, context=CONTEXT, now=1.0),
            v.build_row("j", 1, "good", clip={"predicted_score": 80}, context=CONTEXT, now=1.0),
            v.build_row("j", 2, "bad", "boring", clip={"predicted_score": 70}, context=CONTEXT, now=1.0),
        ]
        out = v.summarise(rows)
        assert out["rated"] == 3
        assert out["by_verdict"] == {"good": 2, "bad": 1}
        assert out["by_reason"] == {"boring": 1}
        assert out["by_mode"] == {"classical": 3}
        # The number that says whether the picker's own score tracks the user.
        assert out["score_split"]["good"] == 85.0
        assert out["score_split"]["bad"] == 70.0

    def test_a_re_rated_clip_is_counted_once(self):
        rows = [v.build_row("j", 0, "good", now=1.0),
                v.build_row("j", 0, "bad", "boring", now=2.0)]
        assert v.summarise(rows)["by_verdict"] == {"bad": 1}

    def test_nothing_rated_yet(self):
        out = v.summarise([])
        assert out["rated"] == 0 and out["score_split"]["good"] is None


class TestUnrating:
    def test_clearing_hides_it_from_the_job_and_the_counts(self):
        # A misclick must be reversible, and "no opinion" is not "bad".
        rows = [v.build_row("j", 0, "good", now=1.0),
                v.build_row("j", 0, "unrated", now=2.0),
                v.build_row("j", 1, "bad", "boring", now=1.0)]
        assert set(v.for_job(rows, "j")) == {1}
        out = v.summarise(rows)
        assert out["rated"] == 1 and out["by_verdict"] == {"bad": 1}

    def test_clearing_drops_any_reason(self):
        assert v.normalize("unrated", "boring") == ("unrated", None)

    def test_the_history_survives_the_clear(self):
        # Append-only: clearing adds a row, it does not erase what was thought.
        rows = [v.build_row("j", 0, "good", now=1.0),
                v.build_row("j", 0, "unrated", now=2.0)]
        assert [r["verdict"] for r in rows] == ["good", "unrated"]

    def test_rating_again_after_clearing_works(self):
        rows = [v.build_row("j", 0, "good", now=1.0),
                v.build_row("j", 0, "unrated", now=2.0),
                v.build_row("j", 0, "bad", "bad_cut", now=3.0)]
        assert v.for_job(rows, "j")[0]["verdict"] == "bad"


class TestErasure:
    def test_drops_only_that_users_rows(self):
        rows = [v.build_row("j", 0, "good", user="u1", now=1.0),
                v.build_row("j", 1, "good", user="u2", now=1.0),
                v.build_row("j", 2, "good", now=1.0)]          # self-host, no user
        kept = v.drop_user(rows, "u1")
        assert [r["clip_index"] for r in kept] == [1, 2]

    def test_rewrite_replaces_the_store(self, tmp_path):
        d = str(tmp_path)
        v.append_row(d, v.build_row("j", 0, "good", user="u1", now=1.0))
        v.append_row(d, v.build_row("j", 1, "good", user="u2", now=1.0))
        v.rewrite(d, v.drop_user(v.read_rows(d), "u1"))
        rows = v.read_rows(d)
        assert len(rows) == 1 and rows[0]["user"] == "u2"

    def test_rewrite_leaves_valid_jsonl(self, tmp_path):
        d = str(tmp_path)
        v.rewrite(d, [v.build_row("j", 0, "good", now=1.0)])
        with open(v.store_path(d), encoding="utf-8") as fh:
            assert all(json.loads(line) for line in fh if line.strip())

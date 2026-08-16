import numpy as np

from src.nlp.frontend import (
    curated_topic_frontend_payload,
    related_episode_rows,
    topic_frontend_payload,
)


def test_related_episodes_use_cosine_similarity_and_exclude_self():
    rows = related_episode_rows(
        np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32),
        [10, 20, 30],
        top_k=1,
    )

    assert len(rows) == 3
    assert next(row for row in rows if row["sr_episode_id"] == 10)["related_episode_id"] == 20
    assert all(row["sr_episode_id"] != row["related_episode_id"] for row in rows)


def test_topic_frontend_payload_omits_outliers_and_keeps_compact_episode_data():
    payload = topic_frontend_payload(
        [
            {"topic_id": -1, "label": "Outliers", "is_outlier": True},
            {"topic_id": 1, "label": "app play", "is_low_quality": True},
            {
                "topic_id": 2,
                "label": "2_familj_barn",
                "keywords": ["familj", "barn", "föräldraskap"],
                "episode_count": 2,
                "chunk_count": 3,
            },
        ],
        [
            {"sr_episode_id": 10, "topic_id": -1, "share": 0.7},
            {"sr_episode_id": 10, "topic_id": 2, "share": 0.3},
        ],
        [{"sr_episode_id": 10, "x": 1.234567, "y": 2.0, "dominant_topic_id": -1}],
        [
            {
                "sr_episode_id": 10,
                "related_episode_id": 20,
                "similarity": 0.87654,
                "rank": 1,
            }
        ],
    )

    assert payload["topics"] == {
        "2": {
            "label": "familj · barn · föräldraskap",
            "keywords": ["familj", "barn", "föräldraskap"],
            "episodes": 2,
            "chunks": 3,
        }
    }
    assert payload["episodes"]["10"] == {
        "x": 1.23457,
        "y": 2.0,
        "dominant": 2,
        "topics": [[2, 0.3]],
        "related": [[20, 0.8765]],
    }


def test_curated_topic_frontend_payload_preserves_string_ids_and_coverage():
    payload = curated_topic_frontend_payload(
        [
            {
                "topic_id": "topic-a",
                "label": "Topic A",
                "parent": "Tema",
                "episodes": 1,
                "chunks": 2,
            }
        ],
        [
            {
                "sr_episode_id": 10,
                "topic_id": "topic-a",
                "coverage": 0.12345,
            }
        ],
        [{"sr_episode_id": 10, "x": 1, "y": 2}],
        [],
    )

    assert payload["topics"]["topic-a"]["parent"] == "Tema"
    assert payload["episodes"]["10"]["topics"] == [["topic-a", 0.1235]]
    assert payload["episodes"]["10"]["dominant"] == "topic-a"


def test_curated_topic_frontend_payload_recounts_topics_from_all_matches():
    payload = curated_topic_frontend_payload(
        [
            {"topic_id": "strong", "label": "Strong", "episodes": 2, "chunks": 1},
            {"topic_id": "weak", "label": "Weak", "episodes": 2, "chunks": 1},
        ],
        [
            {"sr_episode_id": 1, "topic_id": "strong", "share": 0.08},
            {"sr_episode_id": 1, "topic_id": "weak", "share": 0.079},
            {"sr_episode_id": 2, "topic_id": "weak", "share": 0.2},
        ],
        [{"sr_episode_id": 1, "x": 0, "y": 0}, {"sr_episode_id": 2, "x": 0, "y": 0}],
        [],
    )

    assert payload["episodes"]["1"]["topics"] == [["strong", 0.08], ["weak", 0.079]]
    assert payload["topics"]["strong"]["episodes"] == 1
    assert payload["topics"]["weak"]["episodes"] == 2

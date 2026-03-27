import unittest

import numpy as np
import pandas as pd

from scripts import baselines, cli_recommend, collab_filtering, neumf, prepare_data, ranking_eval


class PipelineAndMetricsTests(unittest.TestCase):
    def test_time_aware_split_uses_per_user_80_10_10_blocks(self):
        rows = []
        for ts in range(20):
            rows.append({"userId": 1, "movieId": 100 + ts, "rating": 4.0, "timestamp": ts})
        ratings = pd.DataFrame(rows)

        train_df, val_df, test_df = prepare_data._time_aware_split(ratings)

        self.assertEqual(len(train_df), 16)
        self.assertEqual(len(val_df), 2)
        self.assertEqual(len(test_df), 2)
        self.assertEqual(train_df["timestamp"].tolist(), list(range(16)))
        self.assertEqual(val_df["timestamp"].tolist(), [16, 17])
        self.assertEqual(test_df["timestamp"].tolist(), [18, 19])

    def test_validate_split_outputs_rejects_seen_negatives(self):
        train_df = pd.DataFrame(
            {"user_index": [0], "movie_index": [0], "rating": [4.0], "timestamp": [1]},
        )
        val_df = pd.DataFrame(
            {"user_index": [0], "movie_index": [1], "rating": [5.0], "timestamp": [2]},
        )
        test_df = pd.DataFrame(
            {"user_index": [0], "movie_index": [2], "rating": [3.0], "timestamp": [3]},
        )
        ratings_clean_out = pd.concat([train_df, val_df, test_df], ignore_index=True)
        val_negs_df = pd.DataFrame({"user_index": [0], "pos_movie_index": [1], "neg_movie_index": [2]})
        test_negs_df = pd.DataFrame({"user_index": [0], "pos_movie_index": [2], "neg_movie_index": [0]})

        with self.assertRaisesRegex(ValueError, "overlap with known user history"):
            prepare_data._validate_split_outputs(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                ratings_clean_out=ratings_clean_out,
                val_negs_df=val_negs_df,
                test_negs_df=test_negs_df,
                num_negatives=1,
            )

    def test_topn_metrics_group_by_event_not_user(self):
        candidate_df = pd.DataFrame(
            {
                "event_id": [0, 0, 1, 1],
                "user_index": [7, 7, 7, 7],
                "pos_movie_index": [10, 10, 20, 20],
                "neg_movie_index": [30, 31, 40, 41],
            }
        )

        def score_fn(user_index, candidates):
            if int(candidates[0]) == 10:
                return np.asarray([5.0, 1.0, 0.5], dtype=np.float32)
            return np.asarray([0.2, 4.0, 3.0], dtype=np.float32)

        metrics = baselines.evaluate_topn_candidate_groups(candidate_df, score_fn, ks=[1, 3])

        self.assertAlmostEqual(metrics["recall@1"], 0.5)
        self.assertAlmostEqual(metrics["precision@1"], 0.5)
        self.assertAlmostEqual(metrics["recall@3"], 1.0)

    def test_build_candidate_groups_excludes_full_known_history(self):
        split_df = pd.DataFrame({"user_index": [0], "movie_index": [2]})
        rated_source_df = pd.DataFrame(
            {
                "user_index": [0, 0, 0],
                "movie_index": [0, 1, 2],
            }
        )

        _events_df, negs_df, _summary = ranking_eval.build_candidate_groups(
            split_df=split_df,
            rated_source_df=rated_source_df,
            num_movies=6,
            num_negatives=3,
            seed=7,
            max_events=None,
        )

        self.assertEqual(len(negs_df), 3)
        self.assertTrue(set(negs_df["neg_movie_index"].tolist()).issubset({3, 4, 5}))

    def test_sparse_cold_start_switches_to_metadata_fallback(self):
        self.assertTrue(cli_recommend._should_use_metadata_fallback(2, 3))
        self.assertFalse(cli_recommend._should_use_metadata_fallback(3, 3))
        self.assertFalse(cli_recommend._should_use_metadata_fallback(2, None))

    def test_baselines_final_training_uses_train_plus_val(self):
        train_df = pd.DataFrame({"user_index": [0, 1], "movie_index": [10, 11], "rating": [4.0, 3.5]})
        val_df = pd.DataFrame({"user_index": [2], "movie_index": [12], "rating": [5.0]})

        trainval_df = baselines.build_final_train_df(train_df, val_df)

        self.assertEqual(len(trainval_df), 3)
        self.assertEqual(trainval_df["movie_index"].tolist(), [10, 11, 12])

    def test_als_stage_test_metric_policy(self):
        self.assertFalse(collab_filtering.stage_compute_test_metrics("als_tune"))
        self.assertTrue(collab_filtering.stage_compute_test_metrics("als_final"))
        with self.assertRaisesRegex(ValueError, "Unsupported ALS stage"):
            collab_filtering.stage_compute_test_metrics("knn_tune")

    def test_neumf_filters_training_positives_by_threshold(self):
        ratings_df = pd.DataFrame(
            {
                "user_index": [0, 0, 1, 1],
                "movie_index": [10, 11, 12, 13],
                "rating": [5.0, 3.5, 4.0, 2.0],
                "timestamp": [1, 2, 3, 4],
            }
        )

        positives = neumf.filter_positive_interactions(ratings_df, positive_threshold=4.0)

        self.assertEqual(positives["movie_index"].tolist(), [10, 12])
        self.assertTrue((positives["rating"] >= 4.0).all())


if __name__ == "__main__":
    unittest.main()

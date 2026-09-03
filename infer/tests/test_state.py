import random
import unittest
from unittest.mock import patch

from infer.state import (
    AppendIntent,
    HistoryPool,
    StateManager,
    StateSlotPool,
)


def require_reservation(
    manager: StateManager,
    step_id: int,
    request_id: int,
    max_tokens: int,
    *,
    lane: int | None = None,
    prefix=None,
    cache_token_ids: tuple[int, ...] = (),
):
    plan = manager.reserve(
        step_id,
        (
            AppendIntent(
                request_id,
                max_tokens,
                lane=lane,
                prefix=prefix,
                cache_token_ids=cache_token_ids,
            ),
        ),
    )
    if plan is None:
        raise AssertionError("reservation unexpectedly failed")
    return plan


def acquire_candidate(manager: StateManager, token_ids: tuple[int, ...], lane: int = 0):
    (candidate,) = manager.acquire_prefix_candidates(token_ids, (lane,))
    return candidate


def cached_prefix_length(
    manager: StateManager, token_ids: tuple[int, ...], lane: int = 0
) -> int:
    lease = acquire_candidate(manager, token_ids, lane).lease
    if lease is None:
        return 0
    manager.release_prefix(lease)
    return len(lease.token_ids)


class HistoryPoolTest(unittest.TestCase):
    def test_shares_by_refcount_and_rejects_a_reused_generation(self) -> None:
        pool = HistoryPool(1)
        (first,) = pool.allocate(1) or ()
        first_physical = pool.physical((first,))
        pool.retain((first,))
        pool.release((first,))
        self.assertEqual(pool.available, 0)
        pool.release((first,))
        self.assertEqual(pool.available, 1)

        (second,) = pool.allocate(1) or ()
        self.assertEqual(first_physical, pool.physical((second,)))
        self.assertNotEqual(first, second)
        with self.assertRaisesRegex(ValueError, "stale history block"):
            pool.retain((first,))

    def test_capacity_failure_and_duplicate_release_are_atomic(self) -> None:
        pool = HistoryPool(1)
        (block,) = pool.allocate(1) or ()
        self.assertIsNone(pool.allocate(1))
        with self.assertRaisesRegex(ValueError, "more than once"):
            pool.release((block, block))
        self.assertEqual(pool.available, 0)
        for count in (-1, 1.0):
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                pool.allocate(count)  # type: ignore[arg-type]

    def test_finds_the_shortest_lru_prefix_that_reclaims_enough_blocks(self) -> None:
        pool = HistoryPool(2)
        first, second = pool.allocate(2) or ()
        pool.retain((first,))

        self.assertEqual(
            pool.reclaim_prefix_count(((first,), (first, second)), 1),
            2,
        )
        self.assertEqual(
            pool.reclaim_prefix_count(((first, second), (first,)), 1),
            1,
        )
        pool.retain((first,))
        self.assertIsNone(pool.reclaim_prefix_count(((first,), (first, second)), 2))
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            pool.reclaim_prefix_count((), -1)


class StateSlotPoolTest(unittest.TestCase):
    def test_live_and_snapshot_quotas_do_not_borrow(self) -> None:
        pool = StateSlotPool(live_count=1, snapshot_count=2)
        live = pool.acquire_live()
        first_snapshot = pool.acquire_snapshot()
        second_snapshot = pool.acquire_snapshot()

        self.assertEqual(live, 0)
        self.assertEqual((first_snapshot, second_snapshot), (1, 2))
        self.assertIsNone(pool.acquire_live())
        self.assertIsNone(pool.acquire_snapshot())
        assert first_snapshot is not None
        pool.release(first_snapshot)
        self.assertIsNone(pool.acquire_live())
        self.assertEqual(pool.acquire_snapshot(), first_snapshot)
        with self.assertRaisesRegex(ValueError, "stale state slot"):
            pool.release(99)


class StateManagerTest(unittest.TestCase):
    def test_history_block_geometry_is_configurable(self) -> None:
        manager = StateManager(2, 1, 0, history_block_tokens=4)
        plan = require_reservation(manager, 0, 1, 5, lane=0)

        self.assertEqual(len(plan.rows[0].table_delta.physical_blocks), 2)
        (lease,) = manager.commit(plan, (5,))
        assert lease is not None
        self.assertEqual(len(lease.history_blocks), 2)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            StateManager(1, 1, 0, history_block_tokens=0)

    def test_incremental_reservations_cross_127_128_129_boundaries(self) -> None:
        manager = StateManager(2, 1, 0, history_block_tokens=128)
        first = require_reservation(manager, 0, 1, 127, lane=0)
        self.assertEqual(first.rows[0].table_delta.start_block, 0)
        self.assertEqual(first.rows[0].table_delta.physical_blocks, (0,))
        (at_127,) = manager.commit(first, (127,))
        assert at_127 is not None
        self.assertEqual(len(at_127.history_blocks), 1)

        second = require_reservation(manager, 1, 1, 1)
        self.assertEqual(second.rows[0].table_delta.physical_blocks, ())
        (at_128,) = manager.commit(second, (1,))
        assert at_128 is not None
        self.assertEqual(len(at_128.history_blocks), 1)

        third = require_reservation(manager, 2, 1, 1)
        self.assertEqual(third.rows[0].table_delta.start_block, 1)
        self.assertEqual(third.rows[0].table_delta.physical_blocks, (1,))
        (at_129,) = manager.commit(third, (1,))
        assert at_129 is not None
        self.assertEqual(len(at_129.history_blocks), 2)

    def test_two_ordered_reservations_keep_only_the_accepted_prefix(self) -> None:
        manager = StateManager(3, 2, 0, lane_count=2, history_block_tokens=128)
        first = require_reservation(manager, 0, 1, 129, lane=0)
        second = require_reservation(manager, 1, 1, 129)

        self.assertEqual(first.rows[0].table_delta.physical_blocks, (0, 1))
        self.assertEqual(second.rows[0].table_delta.physical_blocks, (2,))
        with self.assertRaisesRegex(RuntimeError, "outstanding plans"):
            manager.release(first.rows[0].sequence)
        with self.assertRaisesRegex(RuntimeError, "two plans"):
            manager.reserve(2, (AppendIntent(2, 1, lane=1),))
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            manager.commit(second, (1,))

        (after_first,) = manager.commit(first, (1,))
        assert after_first is not None
        self.assertEqual(after_first.committed_tokens, 1)
        self.assertEqual(manager.histories[0].available, 0)
        other_lane = manager.reserve(2, (AppendIntent(2, 1, lane=1),))
        self.assertIsNotNone(other_lane)

        (after_second,) = manager.commit(second, (128,))
        assert after_second is not None
        self.assertEqual(after_second.committed_tokens, 129)
        self.assertEqual(len(after_second.history_blocks), 2)
        self.assertEqual(manager.histories[0].available, 1)
        with self.assertRaisesRegex(ValueError, "stale lease"):
            manager.release(after_first)
        assert other_lane is not None
        (other,) = manager.commit(other_lane, (0,))
        assert other is not None
        manager.release(other)

    def test_cancelled_two_plan_restore_releases_private_suffixes(self) -> None:
        manager = StateManager(5, 2, 1, history_block_tokens=128)
        token_ids = tuple(range(256))
        seed = require_reservation(
            manager,
            0,
            1,
            len(token_ids),
            lane=0,
            cache_token_ids=token_ids,
        )
        (source,) = manager.commit(seed, (len(token_ids),))
        assert source is not None
        manager.release(source)

        prefix = acquire_candidate(manager, token_ids + (999,)).lease
        assert prefix is not None
        first = require_reservation(manager, 1, 2, 129, lane=0, prefix=prefix)
        second = require_reservation(manager, 2, 2, 129)
        shared = manager.histories[0].physical(prefix.shared_history_blocks)[0]
        self.assertEqual(manager.histories[0]._references[shared], 2)
        manager.cancel(2)

        (after_first,) = manager.commit(first, (1,))
        assert after_first is not None
        self.assertEqual(after_first.committed_tokens, 257)
        self.assertEqual(manager.histories[0].available, 0)

        (cancelled,) = manager.commit(second, (0,))
        self.assertIsNone(cancelled)
        self.assertEqual(manager.histories[0].available, 4)
        self.assertEqual(manager.reservable_history_blocks(0), 5)
        self.assertEqual(manager.slots[0].available_live, 2)

    def test_cached_prefix_restores_a_private_tail_block(self) -> None:
        manager = StateManager(4, 2, 1, lane_count=2, history_block_tokens=128)
        token_ids = tuple(range(256))
        first = require_reservation(
            manager,
            0,
            1,
            len(token_ids),
            lane=0,
            cache_token_ids=token_ids,
        )
        self.assertEqual(first.rows[0].capture_slot, 2)
        (source,) = manager.commit(first, (len(token_ids),))
        assert source is not None
        manager.release(source)
        self.assertEqual(manager.histories[0].available, 3)

        prefix = acquire_candidate(manager, token_ids + (999,)).lease
        assert prefix is not None
        self.assertEqual(len(prefix.token_ids), 256)
        self.assertEqual(len(prefix.shared_history_blocks), 1)
        with self.assertRaisesRegex(ValueError, "another lane"):
            manager.reserve(
                1,
                (AppendIntent(2, 1, lane=1, prefix=prefix),),
            )
        second = require_reservation(
            manager,
            1,
            2,
            1,
            lane=0,
            prefix=prefix,
        )
        self.assertEqual(second.rows[0].restore_slot, 2)
        self.assertEqual(second.rows[0].table_delta.start_block, 0)
        self.assertEqual(second.rows[0].table_delta.physical_blocks, (0, 1, 2))
        (restored,) = manager.commit(second, (1,))
        assert restored is not None
        self.assertEqual(restored.history_blocks[:1], prefix.shared_history_blocks)
        self.assertEqual(len(restored.history_blocks), 3)
        manager.release(restored)

    def test_prefix_probe_touches_lru_only_after_transfer(self) -> None:
        manager = StateManager(4, 2, 2, history_block_tokens=128)
        token_sets = (tuple(range(128)), tuple(range(1_000, 1_128)))
        for step_id, token_ids in enumerate(token_sets):
            plan = require_reservation(
                manager,
                step_id,
                step_id + 1,
                len(token_ids),
                lane=0,
                cache_token_ids=token_ids,
            )
            (source,) = manager.commit(plan, (len(token_ids),))
            assert source is not None
            manager.release(source)

        prefix = acquire_candidate(manager, token_sets[0] + (999,)).lease
        assert prefix is not None
        key = (prefix.lane, prefix.digest)
        self.assertEqual(next(iter(manager._prefixes)), key)
        restored = require_reservation(
            manager,
            2,
            3,
            1,
            lane=0,
            prefix=prefix,
        )
        self.assertEqual(next(reversed(manager._prefixes)), key)
        (source,) = manager.commit(restored, (1,))
        assert source is not None
        manager.release(source)

    def test_cache_boundaries_replay_one_complete_block_and_promote_branches(
        self,
    ) -> None:
        manager = StateManager(8, 3, 2, history_block_tokens=128)
        self.assertEqual(
            acquire_candidate(manager, tuple(range(128))).cache_boundaries,
            (),
        )
        self.assertEqual(
            acquire_candidate(manager, tuple(range(129))).cache_boundaries,
            (128,),
        )
        self.assertEqual(
            acquire_candidate(manager, tuple(range(256))).cache_boundaries,
            (128,),
        )
        self.assertEqual(
            acquire_candidate(manager, tuple(range(257))).cache_boundaries,
            (256,),
        )

        shared = tuple(range(128))
        first_tokens = shared + tuple(range(1_000, 1_128))
        first = require_reservation(
            manager,
            0,
            1,
            len(first_tokens),
            lane=0,
            cache_token_ids=first_tokens,
        )
        (source,) = manager.commit(first, (len(first_tokens),))
        assert source is not None
        manager.release(source)

        branch = shared + tuple(range(2_000, 2_129))
        candidate = acquire_candidate(manager, branch)
        self.assertIsNone(candidate.lease)
        self.assertEqual(candidate.promotion_tokens, 128)
        self.assertEqual(candidate.cache_boundaries, (128, 256))

    def test_promotion_index_tracks_shared_ancestors_until_last_eviction(self) -> None:
        manager = StateManager(12, 3, 2, lane_count=2, history_block_tokens=128)
        shared = tuple(range(128))

        for step_id, token_base in enumerate((1_000, 2_000)):
            token_ids = shared + tuple(range(token_base, token_base + 128))
            plan = require_reservation(
                manager,
                step_id,
                step_id + 1,
                len(token_ids),
                lane=0,
                cache_token_ids=token_ids,
            )
            (source,) = manager.commit(plan, (len(token_ids),))
            assert source is not None
            manager.release(source)

        branch = shared + tuple(range(3_000, 3_129))
        self.assertEqual(acquire_candidate(manager, shared).promotion_tokens, 0)
        self.assertEqual(acquire_candidate(manager, branch).promotion_tokens, 128)
        self.assertEqual(
            acquire_candidate(manager, branch, lane=1).promotion_tokens,
            0,
        )

        unrelated = tuple(range(4_000, 4_256))
        replacement = require_reservation(
            manager,
            2,
            3,
            len(unrelated),
            lane=0,
            cache_token_ids=unrelated,
        )
        (source,) = manager.commit(replacement, (len(unrelated),))
        assert source is not None
        manager.release(source)

        self.assertEqual(acquire_candidate(manager, branch).promotion_tokens, 128)
        pressure = require_reservation(manager, 3, 4, 1_536, lane=0)
        self.assertEqual(acquire_candidate(manager, branch).promotion_tokens, 0)
        (source,) = manager.commit(pressure, (1,))
        assert source is not None
        manager.release(source)

    def test_promotion_rejects_a_digest_collision(self) -> None:
        class Collision:
            def digest(self):
                return bytes(16)

        manager = StateManager(4, 1, 1, history_block_tokens=128)
        token_ids = tuple(range(256))
        with patch("infer.state.blake2b", return_value=Collision()):
            plan = require_reservation(
                manager,
                0,
                1,
                len(token_ids),
                lane=0,
                cache_token_ids=token_ids,
            )
            (source,) = manager.commit(plan, (len(token_ids),))
            assert source is not None
            manager.release(source)

            unrelated = tuple(range(1_000, 1_257))
            self.assertEqual(
                acquire_candidate(manager, unrelated).promotion_tokens,
                0,
            )

    def test_capture_repeats_an_already_allocated_tail_block_in_the_delta(
        self,
    ) -> None:
        manager = StateManager(2, 1, 1, history_block_tokens=128)
        first = require_reservation(manager, 0, 1, 1, lane=0)
        manager.commit(first, (1,))

        token_ids = tuple(range(128))
        capture = require_reservation(
            manager,
            1,
            1,
            127,
            cache_token_ids=token_ids,
        )

        self.assertIsNotNone(capture.rows[0].capture_slot)
        self.assertEqual(capture.rows[0].table_delta.start_block, 0)
        self.assertEqual(capture.rows[0].table_delta.physical_blocks, (0,))
        (lease,) = manager.commit(capture, (127,))
        assert lease is not None
        manager.release(lease)

    def test_pinned_prefix_is_not_evicted_for_a_new_capture(self) -> None:
        manager = StateManager(4, 2, 1, history_block_tokens=128)
        first_tokens = tuple(range(128))
        first = require_reservation(
            manager,
            0,
            1,
            128,
            lane=0,
            cache_token_ids=first_tokens,
        )
        (source,) = manager.commit(first, (128,))
        assert source is not None
        manager.release(source)
        prefix = acquire_candidate(manager, first_tokens + (999,)).lease
        assert prefix is not None

        second_tokens = tuple(range(1_000, 1_128))
        second = require_reservation(
            manager,
            1,
            2,
            128,
            lane=0,
            cache_token_ids=second_tokens,
        )
        self.assertIsNone(second.rows[0].capture_slot)
        (uncached,) = manager.commit(second, (128,))
        assert uncached is not None
        manager.release(uncached)
        manager.release_prefix(prefix)
        third_tokens = tuple(range(2_000, 2_128))
        third = require_reservation(
            manager,
            2,
            3,
            128,
            lane=0,
            cache_token_ids=third_tokens,
        )
        self.assertIsNotNone(third.rows[0].capture_slot)
        (replacement,) = manager.commit(third, (128,))
        assert replacement is not None
        manager.release(replacement)
        self.assertEqual(cached_prefix_length(manager, first_tokens + (999,)), 0)
        self.assertEqual(cached_prefix_length(manager, third_tokens + (999,)), 128)

    def test_one_prefix_pin_cannot_back_two_reservations(self) -> None:
        manager = StateManager(4, 3, 1, history_block_tokens=128)
        token_ids = tuple(range(128))
        first = require_reservation(
            manager,
            0,
            1,
            128,
            lane=0,
            cache_token_ids=token_ids,
        )
        (source,) = manager.commit(first, (128,))
        assert source is not None
        manager.release(source)
        prefix = acquire_candidate(manager, token_ids + (999,)).lease
        assert prefix is not None

        with self.assertRaisesRegex(ValueError, "stale prefix lease"):
            manager.reserve(
                1,
                (
                    AppendIntent(2, 1, lane=0, prefix=prefix),
                    AppendIntent(3, 1, lane=0, prefix=prefix),
                ),
            )

        manager.release_prefix(prefix)
        self.assertEqual(manager.slots[0].available_live, 3)

    def test_successful_reserve_consumes_the_prefix_ticket(self) -> None:
        manager = StateManager(4, 2, 1, history_block_tokens=128)
        token_ids = tuple(range(128))
        first = require_reservation(
            manager,
            0,
            1,
            128,
            lane=0,
            cache_token_ids=token_ids,
        )
        (source,) = manager.commit(first, (128,))
        assert source is not None
        manager.release(source)
        prefix = acquire_candidate(manager, token_ids + (999,)).lease
        assert prefix is not None
        restored = require_reservation(
            manager,
            1,
            2,
            1,
            lane=0,
            prefix=prefix,
        )

        with self.assertRaisesRegex(ValueError, "stale prefix lease"):
            manager.release_prefix(prefix)
        uncached = require_reservation(
            manager,
            2,
            3,
            128,
            lane=0,
            cache_token_ids=tuple(range(1_000, 1_128)),
        )
        self.assertIsNone(uncached.rows[0].capture_slot)

        (hit,) = manager.commit(restored, (1,))
        (cold,) = manager.commit(uncached, (128,))
        assert hit is not None and cold is not None
        manager.release(hit)
        manager.release(cold)

    def test_capture_key_must_extend_the_restored_prefix(self) -> None:
        manager = StateManager(4, 2, 2, history_block_tokens=128)
        token_ids = tuple(range(128))
        first = require_reservation(
            manager,
            0,
            1,
            128,
            lane=0,
            cache_token_ids=token_ids,
        )
        (source,) = manager.commit(first, (128,))
        assert source is not None
        manager.release(source)
        prefix = acquire_candidate(manager, token_ids + (999,)).lease
        assert prefix is not None

        with self.assertRaisesRegex(ValueError, "extend the restored prefix"):
            manager.reserve(
                1,
                (
                    AppendIntent(
                        2,
                        128,
                        lane=0,
                        prefix=prefix,
                        cache_token_ids=tuple(range(1_000, 1_256)),
                    ),
                ),
            )

        manager.release_prefix(prefix)
        self.assertEqual(manager.histories[0].available, 4)

    def test_history_pressure_keeps_snapshot_only_prefixes(self) -> None:
        manager = StateManager(2, 2, 2, history_block_tokens=128)
        short_tokens = tuple(range(128))
        short = require_reservation(
            manager,
            0,
            1,
            128,
            lane=0,
            cache_token_ids=short_tokens,
        )
        (short_source,) = manager.commit(short, (128,))
        assert short_source is not None
        manager.release(short_source)

        long_tokens = tuple(range(1_000, 1_256))
        long = require_reservation(
            manager,
            1,
            2,
            256,
            lane=0,
            cache_token_ids=long_tokens,
        )
        (long_source,) = manager.commit(long, (256,))
        assert long_source is not None
        manager.release(long_source)

        replacement = require_reservation(manager, 2, 3, 129, lane=0)
        self.assertEqual(cached_prefix_length(manager, short_tokens + (999,)), 128)
        self.assertEqual(cached_prefix_length(manager, long_tokens + (999,)), 0)
        self.assertEqual(manager.slots[0].available_snapshots, 1)
        (lease,) = manager.commit(replacement, (129,))
        assert lease is not None
        manager.release(lease)
        self.assertEqual(manager.histories[0].available, 2)

    def test_invalid_cached_tokens_fail_before_allocating_state(self) -> None:
        manager = StateManager(2, 1, 1, history_block_tokens=128)
        invalid = tuple(range(127)) + (-1,)

        with self.assertRaisesRegex(ValueError, "unsigned 32-bit"):
            require_reservation(
                manager,
                0,
                1,
                128,
                lane=0,
                cache_token_ids=invalid,
            )

        self.assertEqual(manager.histories[0].available, 2)
        self.assertEqual(manager.slots[0].available_live, 1)
        self.assertEqual(manager.slots[0].available_snapshots, 1)

    def test_failed_multilane_reclaim_does_not_apply_an_earlier_lane_plan(
        self,
    ) -> None:
        manager = StateManager(2, 2, 1, lane_count=2, history_block_tokens=128)
        token_ids = tuple(range(256))
        cached = require_reservation(
            manager,
            0,
            1,
            len(token_ids),
            lane=0,
            cache_token_ids=token_ids,
        )
        (source,) = manager.commit(cached, (len(token_ids),))
        assert source is not None
        manager.release(source)

        occupied = require_reservation(manager, 1, 2, 129, lane=1)
        (blocker,) = manager.commit(occupied, (129,))
        assert blocker is not None

        self.assertIsNone(
            manager.reserve(
                2,
                (
                    AppendIntent(3, 129, lane=0),
                    AppendIntent(4, 1, lane=1),
                ),
            )
        )
        self.assertEqual(cached_prefix_length(manager, token_ids + (999,)), 256)
        self.assertEqual(manager.histories[0].available, 1)

        manager.release(blocker)

    def test_cancel_defers_reclamation_until_both_plans_commit(self) -> None:
        manager = StateManager(2, 1, 0, history_block_tokens=128)
        first = require_reservation(manager, 0, 1, 129, lane=0)
        second = require_reservation(manager, 1, 1, 1)
        first_handle = first.rows[0].table_delta.physical_blocks
        manager.cancel(1)
        manager.cancel(1)

        (intermediate,) = manager.commit(first, (1,))
        self.assertIsNotNone(intermediate)
        self.assertEqual(manager.slots[0].available_live, 0)
        (terminal,) = manager.commit(second, (1,))
        self.assertIsNone(terminal)
        self.assertEqual(manager.histories[0].available, 2)
        self.assertEqual(manager.slots[0].available_live, 1)

        replacement = require_reservation(manager, 2, 2, 129, lane=0)
        self.assertEqual(replacement.rows[0].table_delta.physical_blocks, first_handle)

    def test_failed_batch_reservation_and_commit_are_atomic(self) -> None:
        manager = StateManager(2, 2, 0, lane_count=2, history_block_tokens=128)
        self.assertIsNone(
            manager.reserve(
                0,
                (AppendIntent(1, 129, lane=0), AppendIntent(2, 129, lane=0)),
            )
        )
        self.assertEqual(manager.histories[0].available, 2)
        self.assertEqual(manager.slots[0].available_live, 2)

        plan = manager.reserve(
            0,
            (AppendIntent(1, 1, lane=0), AppendIntent(2, 1, lane=1)),
        )
        assert plan is not None
        with self.assertRaisesRegex(ValueError, "exceeds"):
            manager.commit(plan, (1, 2))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            manager.commit(plan, (1, 1.0))  # type: ignore[arg-type]
        leases = manager.commit(plan, (1, 1))
        self.assertEqual([lease.committed_tokens for lease in leases if lease], [1, 1])

    def test_rejects_duplicate_steps_and_non_integer_token_counts(self) -> None:
        manager = StateManager(2, 2, 0, history_block_tokens=128)
        plan = require_reservation(manager, 7, 1, 1, lane=0)
        with self.assertRaisesRegex(ValueError, "already has a reservation"):
            manager.reserve(7, (AppendIntent(2, 1, lane=0),))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            manager.reserve(8, (AppendIntent(2, 1.0, lane=0),))  # type: ignore[arg-type]
        manager.commit(plan, (1,))

    def test_randomized_ownership_cancel_and_two_plan_sequences_do_not_leak(
        self,
    ) -> None:
        rng = random.Random(7)
        manager = StateManager(32, 8, 3, lane_count=4, history_block_tokens=128)
        plans = []
        leases = {}
        pending = {}
        cancelled = set()
        next_request = 0
        next_step = 0

        for _ in range(500):
            if plans and (len(plans) == 2 or rng.random() < 0.45):
                plan = plans.pop(0)
                request_id = plan.rows[0].sequence.request_id
                (lease,) = manager.commit(
                    plan,
                    (rng.randrange(plan.rows[0].max_tokens + 1),),
                )
                pending[request_id] -= 1
                if lease is None:
                    leases.pop(request_id, None)
                    pending.pop(request_id)
                    cancelled.remove(request_id)
                else:
                    leases[request_id] = lease
                continue

            releasable = [request for request in leases if pending[request] == 0]
            if releasable and rng.random() < 0.15:
                request_id = rng.choice(releasable)
                manager.release(leases.pop(request_id))
                pending.pop(request_id)
                continue
            cancellable = [request for request in leases if request not in cancelled]
            if cancellable and rng.random() < 0.15:
                request_id = rng.choice(cancellable)
                manager.cancel(request_id)
                cancelled.add(request_id)
                if pending[request_id] == 0:
                    leases.pop(request_id)
                    pending.pop(request_id)
                    cancelled.remove(request_id)
                continue

            candidates = [
                request
                for request in leases
                if request not in cancelled and pending[request] < 2
            ]
            if candidates and rng.random() < 0.65:
                request_id = rng.choice(candidates)
                intent = AppendIntent(request_id, rng.randint(1, 129))
            else:
                request_id = next_request
                next_request += 1
                intent = AppendIntent(
                    request_id, rng.randint(1, 129), lane=request_id % 4
                )

            plan = manager.reserve(next_step, (intent,))
            if plan is None:
                continue
            next_step += 1
            plans.append(plan)
            leases.setdefault(request_id, plan.rows[0].sequence)
            pending[request_id] = pending.get(request_id, 0) + 1

        for request_id in tuple(leases):
            manager.cancel(request_id)
            cancelled.add(request_id)
        while plans:
            plan = plans.pop(0)
            manager.commit(plan, (0,))
        self.assertTrue(all(pool.available == 32 for pool in manager.histories))
        self.assertTrue(all(pool.available_live == 8 for pool in manager.slots))
        self.assertTrue(all(pool.available_snapshots == 3 for pool in manager.slots))


if __name__ == "__main__":
    unittest.main()

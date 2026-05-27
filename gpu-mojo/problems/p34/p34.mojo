from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext, Dim
from std.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    cluster_arrive,
    cluster_wait,
    elect_one_sync,
)
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.sys import argv
from std.testing import assert_equal, assert_almost_equal, assert_true

comptime SIZE = 1024
comptime TPB = 256
comptime CLUSTER_SIZE = 4
comptime dtype = DType.float32
comptime in_layout = row_major[SIZE]()
comptime out_layout = row_major[1]()
comptime InLayout = type_of(in_layout)
comptime OutLayout = type_of(out_layout)
comptime cluster_layout = row_major[CLUSTER_SIZE]()
comptime ClusterLayout = type_of(cluster_layout)


# ANCHOR: cluster_coordination_basics
def cluster_coordination_basics[
    tpb: Int
](
    output: TileTensor[mut=True, dtype, ClusterLayout, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, InLayout, MutAnyOrigin],
    size: Int,
):
    """Real cluster coordination using SM90+ cluster APIs."""
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

    # Check what's happening with cluster ranks
    var my_block_rank = Int(block_rank_in_cluster())
    var block_id = block_idx.x

    var shared_data = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[tpb]())

    # FIX: Use block_idx.x for data distribution instead of cluster rank
    # Each block should process different portions of the data
    var data_scale = Scalar[dtype](
        block_id + 1
    )  # Use block_idx instead of cluster rank

    # Phase 1: Each block processes its portion
    if global_i < size:
        shared_data[local_i] = input[global_i] * data_scale
    else:
        shared_data[local_i] = 0.0

    barrier()

    # Phase 2: Use cluster_arrive() for inter-block coordination
    # Signal this block has completed processing

    cluster_arrive()

    # Block-level aggregation (only thread 0)
    if local_i == 0:
        var sum: Scalar[dtype] = 0
        for i in range(tpb):
            sum += shared_data[i]
        output[block_id] = sum

    # Wait for all blocks in cluster to complete

    cluster_wait()


# ANCHOR_END: cluster_coordination_basics


# ANCHOR: cluster_collective_operations
def cluster_collective_operations[
    tpb: Int
](
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, InLayout, MutAnyOrigin],
    temp_storage: TileTensor[mut=True, dtype, ClusterLayout, MutAnyOrigin],
    size: Int,
):
    """Cluster-wide collective operations using real cluster APIs."""
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

    var my_block_rank = Int(block_rank_in_cluster())
    var block_id = block_idx.x

    # load data
    var shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[tpb]())

    if global_i < size:
        shared[local_i] = input[global_i]
    else:
        shared[local_i] = 0
    barrier()

    # parallel tree reduction
    var stride = tpb // 2
    while stride > 0:
        if local_i < stride and local_i + stride < tpb:
            shared[local_i] += shared[local_i + stride]
        barrier()
        stride = stride // 2

    if local_i == 0:
        temp_storage[block_id] = shared[0]

    cluster_sync()

    if my_block_rank == 0 and elect_one_sync():
        var sum: Scalar[dtype] = 0
        for i in range(CLUSTER_SIZE):
            sum += temp_storage[i]
        output[0] = sum


# ANCHOR_END: cluster_collective_operations


# ANCHOR: advanced_cluster_patterns
def advanced_cluster_patterns[
    tpb: Int
](
    output: TileTensor[mut=True, dtype, ClusterLayout, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, InLayout, MutAnyOrigin],
    size: Int,
):
    """Advanced cluster programming using cluster masks and relaxed synchronization.
    """
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

    var my_block_rank = Int(block_rank_in_cluster())
    var block_id = block_idx.x

    var shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[tpb]())

    var data_scale = Scalar[dtype](block_id + 1)
    if global_i < size:
        shared[local_i] = input[global_i] * data_scale
    else:
        shared[local_i] = 0
    barrier()

    if elect_one_sync():
        var warp_sum: Scalar[dtype] = 0
        var warp_start = (local_i // 32) * 32
        for i in range(32):
            if warp_start + i < tpb:
                warp_sum += shared[warp_start + i]
        shared[warp_start] = warp_sum
    barrier()

    cluster_arrive()

    if local_i == 0:
        var block_sum: Scalar[dtype] = 0
        for i in range(0, tpb, 32):
            block_sum += shared[i]
        output[block_id] = block_sum

    cluster_wait()


# ANCHOR_END: advanced_cluster_patterns


def main() raises:
    """Test cluster programming concepts using proper Mojo GPU patterns."""
    if len(argv()) < 2:
        print("Usage: p34.mojo [--coordination | --reduction | --advanced]")
        return

    with DeviceContext() as ctx:
        if argv()[1] == "--coordination":
            print("Testing Multi-Block Coordination")
            print("SIZE:", SIZE, "TPB:", TPB, "CLUSTER_SIZE:", CLUSTER_SIZE)

            input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)
            output_buf = ctx.enqueue_create_buffer[dtype](CLUSTER_SIZE)
            output_buf.enqueue_fill(0)

            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    input_host[i] = Scalar[dtype](i % 10) * 0.1

            input_tensor = TileTensor[mut=False, dtype, InLayout](
                input_buf, in_layout
            )
            output_tensor = TileTensor[mut=True, dtype, ClusterLayout](
                output_buf, cluster_layout
            )

            comptime kernel = cluster_coordination_basics[TPB]
            ctx.enqueue_function[kernel](
                output_tensor,
                input_tensor,
                SIZE,
                grid_dim=(CLUSTER_SIZE, 1),
                block_dim=(TPB, 1),
                cluster_dim=Dim(CLUSTER_SIZE, 1, 1),
            )

            ctx.synchronize()

            with output_buf.map_to_host() as result_host:
                print("Block coordination results:")
                for i in range(CLUSTER_SIZE):
                    print("  Block", i, ":", result_host[i])

                # FIX: Verify each block produces NON-ZERO results using proper Mojo testing
                for i in range(CLUSTER_SIZE):
                    assert_true(
                        result_host[i] > 0.0
                    )  # All blocks SHOULD produce non-zero results
                    print("Block", i, "produced result:", result_host[i])

                # FIX: Verify scaling pattern - each block should have DIFFERENT results
                # Due to scaling by block_id + 1 in the kernel
                assert_true(
                    result_host[1] > result_host[0]
                )  # Block 1 > Block 0
                assert_true(
                    result_host[2] > result_host[1]
                )  # Block 2 > Block 1
                assert_true(
                    result_host[3] > result_host[2]
                )  # Block 3 > Block 2
                print("Puzzle 34 complete")

        elif argv()[1] == "--reduction":
            print("Testing Cluster-Wide Reduction")
            print("SIZE:", SIZE, "TPB:", TPB, "CLUSTER_SIZE:", CLUSTER_SIZE)

            input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)
            output_buf = ctx.enqueue_create_buffer[dtype](1)
            output_buf.enqueue_fill(0)
            var temp_buf = ctx.enqueue_create_buffer[dtype](CLUSTER_SIZE)
            temp_buf.enqueue_fill(0)

            var expected_sum: Float32 = 0.0
            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    input_host[i] = Scalar[dtype](i)
                    expected_sum += input_host[i]

            print("Expected sum:", expected_sum)

            input_tensor = TileTensor[mut=False, dtype, InLayout](
                input_buf, in_layout
            )
            var output_tensor = TileTensor[mut=True, dtype, OutLayout](
                output_buf, out_layout
            )
            var temp_tensor = TileTensor[mut=True, dtype, ClusterLayout](
                temp_buf, cluster_layout
            )

            comptime kernel = cluster_collective_operations[TPB]
            ctx.enqueue_function[kernel](
                output_tensor,
                input_tensor,
                temp_tensor,
                SIZE,
                grid_dim=(CLUSTER_SIZE, 1),
                block_dim=(TPB, 1),
                cluster_dim=Dim(CLUSTER_SIZE, 1, 1),
            )

            ctx.synchronize()

            with output_buf.map_to_host() as result_host:
                result = result_host[0]
                print("Cluster reduction result:", result)
                print("Expected:", expected_sum)
                print("Error:", abs(result - expected_sum))

                # Test cluster reduction accuracy with proper tolerance
                assert_almost_equal(
                    result, expected_sum, atol=10.0
                )  # Reasonable tolerance for cluster coordination
                print("Passed: Cluster reduction accuracy test")
                print("Puzzle 34 complete")

        elif argv()[1] == "--advanced":
            print("Testing Advanced Cluster Algorithms")
            print("SIZE:", SIZE, "TPB:", TPB, "CLUSTER_SIZE:", CLUSTER_SIZE)

            input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)
            output_buf = ctx.enqueue_create_buffer[dtype](CLUSTER_SIZE)
            output_buf.enqueue_fill(0)

            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    input_host[i] = (
                        Scalar[dtype](i % 50) * 0.02
                    )  # Pattern for testing

            input_tensor = TileTensor[mut=False, dtype, InLayout](
                input_buf, in_layout
            )
            output_tensor = TileTensor[mut=True, dtype, ClusterLayout](
                output_buf, cluster_layout
            )

            comptime kernel = advanced_cluster_patterns[TPB]
            ctx.enqueue_function[kernel](
                output_tensor,
                input_tensor,
                SIZE,
                grid_dim=(CLUSTER_SIZE, 1),
                block_dim=(TPB, 1),
                cluster_dim=Dim(CLUSTER_SIZE, 1, 1),
            )

            ctx.synchronize()

            with output_buf.map_to_host() as result_host:
                print("Advanced cluster algorithm results:")
                for i in range(CLUSTER_SIZE):
                    print("  Block", i, ":", result_host[i])

                # FIX: Advanced pattern should produce NON-ZERO results
                for i in range(CLUSTER_SIZE):
                    assert_true(
                        result_host[i] > 0.0
                    )  # All blocks SHOULD produce non-zero results
                    print("Advanced Block", i, "result:", result_host[i])

                # FIX: Advanced pattern should show DIFFERENT scaling per block
                assert_true(
                    result_host[1] > result_host[0]
                )  # Block 1 > Block 0
                assert_true(
                    result_host[2] > result_host[1]
                )  # Block 2 > Block 1
                assert_true(
                    result_host[3] > result_host[2]
                )  # Block 3 > Block 2

                print("Puzzle 34 complete")

        else:
            print(
                "Available options: [--coordination | --reduction | --advanced]"
            )

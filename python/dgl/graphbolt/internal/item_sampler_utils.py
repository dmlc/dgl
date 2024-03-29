"""Utility functions for DistributedItemSampler."""


def count_split(total, num_workers, worker_id, batch_size=1):
    """Calculate the number of assigned items after splitting them by batch
    size evenly. It will return the number for this worker and also a sum of
    previous workers.
    """
    quotient, remainder = divmod(total, num_workers * batch_size)
    if batch_size == 1:
        assigned = quotient + (worker_id < remainder)
    else:
        batch_count, last_batch = divmod(remainder, batch_size)
        assigned = quotient * batch_size + (
            batch_size
            if worker_id < batch_count
            else (last_batch if worker_id == batch_count else 0)
        )
    prefix_sum = quotient * worker_id * batch_size + min(
        worker_id * batch_size, remainder
    )
    return (assigned, prefix_sum)


def calculate_range(
    distributed,
    total,
    num_replicas,
    rank,
    num_workers,
    worker_id,
    batch_size,
    drop_last,
    drop_uneven_inputs,
):
    """Calculates the range of items to be assigned to the current worker.

    This function evenly distributes `total` items among multiple workers,
    batching them using `batch_size`. Each replica has `num_workers` workers.
    The batches generated by workers within the same replica are combined into
    the replica`s output. The `drop_last` parameter determines whether
    incomplete batches should be dropped. If `drop_last` is True, incomplete
    batches are discarded. The `drop_uneven_inputs` parameter determines if the
    number of batches assigned to each replica should be the same. If
    `drop_uneven_inputs` is True, excessive batches for some replicas will be
    dropped.

    Args:
        distributed (bool): Whether it's in distributed mode.
        total (int): The total number of items.
        num_replicas (int): The total number of replicas.
        rank (int): The rank of the current replica.
        num_workers (int): The number of workers per replica.
        worker_id (int): The ID of the current worker.
        batch_size (int): The desired batch size.
        drop_last (bool): Whether to drop incomplete batches.
        drop_uneven_inputs (bool): Whether to drop excessive batches for some
          replicas.

    Returns:
        tuple: A tuple containing three numbers:
            - start_offset (int): The starting offset of the range assigned to
              the current worker.
            - assigned_count (int): The length of the range assigned to the
              current worker.
            - output_count (int): The number of items that the current worker
              will produce after dropping.
    """
    # Check if it's distributed mode.
    if not distributed:
        if not drop_last:
            return (0, total, total)
        else:
            return (0, total, total // batch_size * batch_size)
    # First, equally distribute items into all replicas.
    assigned_count, start_offset = count_split(
        total, num_replicas, rank, batch_size
    )
    # Calculate the number of outputs when drop_uneven_inputs is True.
    # `assigned_count` is the number of items distributed to the current
    # process. `output_count` is the number of items should be output
    # by this process after dropping.
    if not drop_uneven_inputs:
        if not drop_last:
            output_count = assigned_count
        else:
            output_count = assigned_count // batch_size * batch_size
    else:
        if not drop_last:
            min_item_count, _ = count_split(
                total, num_replicas, num_replicas - 1, batch_size
            )
            min_batch_count = (min_item_count + batch_size - 1) // batch_size
            output_count = min(min_batch_count * batch_size, assigned_count)
        else:
            output_count = total // (batch_size * num_replicas) * batch_size
    # If there are multiple workers, equally distribute the batches to
    # all workers.
    if num_workers > 1:
        # Equally distribute the dropped number too.
        dropped_items, prev_dropped_items = count_split(
            assigned_count - output_count, num_workers, worker_id
        )
        output_count, prev_output_count = count_split(
            output_count,
            num_workers,
            worker_id,
            batch_size,
        )
        assigned_count = output_count + dropped_items
        start_offset += prev_output_count + prev_dropped_items
    return (start_offset, assigned_count, output_count)

from qdrant_client import QdrantClient
from qdrant_client.models import (NamedSparseVector, SparseIndexParams, SparseVectorParams, OptimizersConfigDiff,
                                  SparseVector)
import click
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
from tqdm import tqdm

from src.sparse_matrix import read_sparse_matrix, knn_result_read
from src.stats import compute_dataset_stats


def csr_to_sparse_vector(point_csr) -> SparseVector:
    indices = point_csr.indices.tolist()
    values = point_csr.data.tolist()
    return SparseVector(indices=indices, values=values)


def insert_generator(vector_name, csr_matrix):
    vec_count = csr_matrix.shape[0]
    for i in range(0, vec_count):
        point = csr_matrix[i]
        vector = csr_to_sparse_vector(point)
        entry = {vector_name: vector}
        yield entry


def longest_posting_list_for_vector(sparse_vector, stats):
    longest = 0
    for index in sparse_vector.indices:
        if index in stats.sizes:
            longest = max(longest, stats.sizes[index])
    return longest


def sum_posting_list_for_vector(sparse_vector, stats):
    total = 0
    for index in sparse_vector.indices:
        if index in stats.sizes:
            total += stats.sizes[index]
    return total


def print_segment_info(host, collection_name):
    # get telemetry
    telemetry = requests.get(f"http://{host}:6333/telemetry?details_level=10").json()
    # make sure that it is the right collection
    collections = telemetry["result"]["collections"]["collections"]
    collection_index = 0
    for i in range(0, len(collections)):
        if collections[i]["id"] == collection_name:
            collection_index = i
            break
    segments = collections[collection_index]["shards"][0]["local"]["segments"]
    print(f"Segments stats:")
    for segment in segments:
        print(f"- {segment['info']['num_points']} points")


@click.command()
@click.option('--host', default="localhost", help="The host of the Qdrant server")
@click.option('--skip-creation', default=True, help="Whether to skip collection creation")
@click.option('--dataset', default="small", help="Dataset to use: small, 1M, full")
@click.option('--slow-ms', default=500, help="Slow query threshold in milliseconds")
@click.option('--search-limit', default=10, help="Search limit")
@click.option('--data-path', default="./data", help="Path to the data files")
@click.option('--segment-number', default=8, help="Number of segments")
@click.option('--analyze-data', default=False, help="Whether to analyze data")
@click.option('--check-groundtruth', default=False, help="Whether to check results against ground truth")
@click.option('--graph-y-limit', default=None, help="Y axis limit for the graph to help compare plots")
@click.option('--parallel-batch-upsert', default=5, help="Number of parallel batch upserts")
@click.option('--on-disk-index', default=False, help="Whether to use on-disk index")
def sparse_vector_benchmark(
        host,
        skip_creation,
        dataset,
        slow_ms,
        search_limit,
        data_path,
        segment_number,
        analyze_data,
        check_groundtruth,
        graph_y_limit,
        parallel_batch_upsert,
        on_disk_index):
    """Sparse vector benchmark tool for Qdrant."""

    collection_name = f"neurIPS_sparse_{dataset}_bench"
    vector_name = "neurIPS"

    # TODO download gzip dataset from server if the file doesn't exist
    if dataset == "small":
        # 100k vectors
        data_file_name = f"{data_path}/base_small.csr"
        gt_file_name = f"{data_path}/base_small.dev.gt"
    elif dataset == "1M":
        # 1M vectors
        data_file_name = f"{data_path}/base_1M.csr"
        gt_file_name = f"{data_path}/base_1M.dev.gt"
    elif dataset == "full":
        # 10M vectors
        data_file_name = f"{data_path}/base_full.csr"
        gt_file_name = f"{data_path}/base_full.dev.gt"
    else:
        print(f"Unknown dataset {dataset}")
        exit(1)

    ground_vectors = []
    if check_groundtruth:
        # https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/dataset_preparation/make_sparse_groundtruth.py
        ground = knn_result_read(gt_file_name)
        print(f"Ground truth contains {ground[0].shape[0]} vectors")
        for i in range(0, ground[0].shape[0]):
            indices = ground[0][i]
            values = ground[1][i]
            assert len(indices) == len(values)
            # sort by index
            sorted_indices = np.argsort(indices)
            sparse_vector = SparseVector(
                indices=indices[sorted_indices],
                values=values[sorted_indices]
            )
            ground_vectors.append(sparse_vector)

    data = {}
    vec_count = 0

    if analyze_data or not skip_creation:
        print(f"Reading {data_file_name} ...")
        data = read_sparse_matrix(data_file_name)
        vec_count = data.shape[0]

    # data analyze behind flag as it can be expensive
    if analyze_data:
        stats = compute_dataset_stats(data)
        dim_count = len(stats.posting_len_per_dimension)
        print(f"Dataset contains {vec_count} sparse vectors with {dim_count} unique dimensions")
        # bar chart of posting list lengths
        plt.scatter(*zip(*stats.count_per_posting_len.items()), )
        plt.grid(True)
        plt.xlabel('Posting list length')
        plt.ylabel('Number of dimensions')
        plt.title(f"Posting list length distribution ({dim_count} dimensions, {vec_count} vectors)")
        plt.savefig(f"./results/neurIPS_bench_{dataset}_posting_len.png")
        plt.close()
        print("Posting list lengths (top 20)")
        # sort by count
        sorted_count_per_index = sorted(stats.posting_len_per_dimension.items(), key=lambda item: item[1], reverse=True)
        for index, count in sorted_count_per_index[:20]:
            # percentage of vectors with this index
            percentage = round(count / vec_count * 100, 2)
            print(f"{index}: {count} ({percentage}%)")

    # gRPC client
    client = QdrantClient(host=host, prefer_grpc=True)
    if not skip_creation:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config={},
            sparse_vectors_config={
                vector_name: SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=on_disk_index,
                    )
                )
            },
            optimizers_config=OptimizersConfigDiff(default_segment_number=segment_number)
        )

        print(f"Uploading {vec_count} sparse vectors into '{collection_name}'")
        client.upload_collection(
            collection_name=collection_name,
            vectors=tqdm(insert_generator(vector_name, data), total=vec_count),
            parallel=parallel_batch_upsert,
            wait=True
        )
        print("Upload done")
    else:
        print("Skipping collection creation")

    # wait for indexing
    print(f"Waiting for collection {collection_name} to index...")
    info = client.get_collection(collection_name=collection_name)
    while info.status != "green":
        print(f"Waiting for collection {collection_name} to index...")
        time.sleep(2)
        info = client.get_collection(collection_name=collection_name)

    # collection stats
    print("Collection is ready for querying:")
    print(f"- {info.points_count} points")
    print(f"- {info.vectors_count} vector count")
    segment_count = info.segments_count
    print(f"- {segment_count} segments")
    print_segment_info(host, collection_name)

    # read queries
    query_data_file_name = f"{data_path}/queries.dev.csr"
    query_data = read_sparse_matrix(query_data_file_name)
    query_count = query_data.shape[0]
    # data for plotting
    latency = []
    dimensions = []
    print("---------------------------------------")
    print(f"Querying {query_count} sparse vectors")
    for i in tqdm(range(0, query_count)):
        try:
            point = query_data[i]
            query_vector = csr_to_sparse_vector(point)
            start = time.time_ns()
            results = client.search(
                collection_name=collection_name,
                with_vectors=check_groundtruth,  # return vector for ground truth check
                with_payload=False,
                limit=search_limit,
                query_vector=NamedSparseVector(
                    name=vector_name,
                    vector=query_vector
                )
            )
            end = time.time_ns()
            duration_ms = (end - start) / 100_000
            latency.append(duration_ms)
            dim = len(query_vector.indices)
            dimensions.append(dim)
            if duration_ms > slow_ms:
                print(f"Slow query with dim {dim} took {duration_ms} millis")
            if check_groundtruth:
                # check results against ground truth
                ground_vector = ground_vectors[i]
                contains = False
                for j in range(0, len(results)):
                    result = results[j].vector[vector_name]
                    if result == ground_vector:
                        contains = True
                        break
                # TODO fix ground truth check
                if not contains:
                    print(f"Results for query {i} doesn't contain the expected groundtruth vector")
                    print(f"Query: {query_vector}")
                    print(f"Ground truth: {ground_vector}")
                    exit(1)
        except KeyboardInterrupt:
            print("Bye - generating partial report")
            # break the loop and generate partial report
            break

    print("---------------------------------------")
    print("Latency distribution:")
    quantiles = np.quantile(latency, [0.5, 0.95, 0.99, 0.999, 1])
    print(f"50p: {round(quantiles[0], 2)} millis")
    print(f"95p: {round(quantiles[1], 2)} millis")
    print(f"99p: {round(quantiles[2], 2)} millis")
    print(f"999p: {round(quantiles[3], 2)} millis")
    print(f"max: {round(quantiles[4], 2)} millis")
    print("")

    # query dimensions distribution
    if analyze_data:
        print("Query dimensions distribution:")
        quantiles = np.quantile(dimensions, [0.5, 0.95, 0.99, 0.999, 1])
        print(f"50p: {quantiles[0]}")
        print(f"95p: {quantiles[1]}")
        print(f"99p: {quantiles[2]}")
        print(f"999p: {quantiles[3]}")
        print(f"max: {quantiles[4]}")

    # Create a 2D histogram of the query dimensions and latencies
    # https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
    timestamp = int(time.time())
    title = f"Sparse NeurIPS {dataset} ({segment_count} segments)"
    plt.hist2d(dimensions, latency, bins=100, cmap="rainbow")
    plt.grid(True)
    # force y axis limits to be able to compare plots
    if graph_y_limit is not None:
        axis = plt.gca()
        axis.set_ylim(bottom=0, top=int(graph_y_limit))
    cbar = plt.colorbar()
    cbar.set_label('Frequency')
    plt.xlabel('Query dimension count')
    plt.ylabel('Latency (ms)')
    plt.title(title)
    plot_file_name = f"./results/sparse_bench_{dataset}_{timestamp}.png"
    print(f"Saving plot to {plot_file_name}")
    plt.savefig(plot_file_name)
    plt.close()

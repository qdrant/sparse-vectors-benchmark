import os

from qdrant_client import QdrantClient
from qdrant_client.models import (NamedSparseVector, SparseIndexParams, SparseVectorParams, OptimizersConfigDiff,
                                  PointStruct, SparseVector)
import click
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
from tqdm import tqdm

from src.download import download_gz_file, download_file
from src.sparse_matrix import read_sparse_matrix, knn_result_read
from src.stats import compute_dataset_stats, compare_floats_percentage


def csr_to_sparse_vector(point_csr) -> SparseVector:
    indices = point_csr.indices.tolist()
    values = point_csr.data.tolist()
    return SparseVector(indices=indices, values=values)


def insert_generator(vector_name, csr_matrix):
    vec_count = csr_matrix.shape[0]
    for i in range(0, vec_count):
        point = csr_matrix[i]
        sparse_vector = csr_to_sparse_vector(point)
        vector = {vector_name: sparse_vector}
        point_struct = PointStruct(id=i, vector=vector)
        yield point_struct


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
    print("Segments stats:")
    for segment in segments:
        print(f"- {segment['info']['num_points']} points")


# query data files
QUERY_DATA_FILE_NAME = "queries.dev.csr"
# small dataset
SMALL_DATA_FILE_NAME = "base_small.csr"
SMALL_GT_FILE_NAME = "base_small.dev.gt"
# 1M dataset
M1_DATA_FILE_NAME = "base_1M.csr"
M1_GT_FILE_NAME = "base_1M.dev.gt"
# full dataset
FULL_DATA_FILE_NAME = "base_full.csr"
FULL_GT_FILE_NAME = "base_full.dev.gt"


def file_names_for_dataset(dataset) -> (str, str):
    if dataset == "small":
        return SMALL_DATA_FILE_NAME, SMALL_GT_FILE_NAME
    elif dataset == "1M":
        return M1_DATA_FILE_NAME, M1_GT_FILE_NAME
    elif dataset == "full":
        return FULL_DATA_FILE_NAME, FULL_GT_FILE_NAME
    else:
        print(f"Unknown dataset {dataset}")
        exit(1)


@click.command()
@click.option('--host', default="localhost", help="The host of the Qdrant server")
@click.option('--skip-creation', default=True, help="Whether to skip collection creation")
@click.option('--dataset', default="small", help="Dataset to use: small, 1M, full")
@click.option('--slow-ms', default=500, help="Slow query threshold in milliseconds")
@click.option('--search-limit', default=10, help="Search limit")
@click.option('--data-path', default="./data", help="Path to the data files")
@click.option('--results-path', default="./results", help="Path to the results files")
@click.option('--segment-number', default=8, help="Number of segments")
@click.option('--analyze-data', default=False, help="Whether to analyze data")
@click.option('--check-ground-truth', default=False, help="Whether to check results against ground truth")
@click.option('--graph-y-range', default=None, help="Y axis range for the graph to help compare plots")
@click.option('--upsert-batch-size', default=512, help="Number of vectors per batch upserts")
@click.option('--parallel-batch-upsert', default=16, help="Number of parallel batch upserts")
@click.option('--on-disk-index', default=False, help="Whether to use on-disk index")
def sparse_vector_benchmark(
        host,
        skip_creation,
        dataset,
        slow_ms,
        search_limit,
        data_path,
        results_path,
        segment_number,
        analyze_data,
        check_ground_truth,
        graph_y_range,
        upsert_batch_size,
        parallel_batch_upsert,
        on_disk_index):
    """Sparse vector benchmark tool for Qdrant."""

    # Make sure working folders exist
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    collection_name = f"neurIPS_sparse_{dataset}_bench"
    vector_name = "neurIPS"

    # pick dataset
    data_file_name, gt_file_name = file_names_for_dataset(dataset)

    # check if files exist
    query_data_file_path = f"{data_path}/{QUERY_DATA_FILE_NAME}"
    if not os.path.isfile(query_data_file_path):
        print(f"Query data file {query_data_file_path} doesn't exist")
        download_gz_file(data_path, QUERY_DATA_FILE_NAME)

    data_file_path = f"{data_path}/{data_file_name}"
    if not os.path.isfile(data_file_path):
        print(f"Data file {data_file_path} doesn't exist")
        download_gz_file(data_path, data_file_name)

    gt_file_path = f"{data_path}/{gt_file_name}"
    if check_ground_truth and not os.path.isfile(gt_file_path):
        print(f"Ground truth file {gt_file_path} doesn't exist")
        download_file(data_path, gt_file_name)

    # ground truth data
    gt_indices = []
    gt_scores = []
    if check_ground_truth:
        # https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/dataset_preparation/make_sparse_groundtruth.py
        gt_indices, gt_scores = knn_result_read(gt_file_path)
        assert len(gt_indices) == len(gt_scores)
        gt_len = len(gt_indices)
        top_len = len(gt_indices[0])
        assert top_len == len(gt_scores[0])
        print(f"Ground truth contains {gt_len} entries for top {top_len} vectors")

    data = {}
    vec_count = 0

    if analyze_data or not skip_creation:
        print(f"Reading {data_file_path} ...")
        data = read_sparse_matrix(data_file_path)
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
    indexing_start = time.time_ns()
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
        client.upload_points(
            collection_name=collection_name,
            points=tqdm(insert_generator(vector_name, data), total=vec_count),
            parallel=parallel_batch_upsert,
            batch_size=upsert_batch_size,
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
        print(f"Collection status: {info.status}")

    indexing_end = time.time_ns()
    indexing_duration_sec = round((indexing_end - indexing_start) / 1_000_000_000, 2)
    print(f"Upload & indexing took {indexing_duration_sec} seconds")

    # collection stats
    print("Collection is ready for querying:")
    print(f"- {info.points_count} points")
    print(f"- {info.vectors_count} vector count")
    segment_count = info.segments_count
    print(f"- {segment_count} segments")
    print_segment_info(host, collection_name)

    # read queries
    query_data = read_sparse_matrix(query_data_file_path)
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
                with_vectors=False,
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
            if check_ground_truth:
                expected_scores = gt_scores[i]
                expected_ids = gt_indices[i]
                # check each result against ground truth
                for j in range(0, len(results)):
                    result = results[j]
                    result_score = result.score
                    expected_score = float(expected_scores[j])
                    # compare floats with 1% tolerance
                    if not compare_floats_percentage(result_score, expected_score, 1):
                        print(f"GT score mismatch vector:{i} result:{j}/{search_limit}: {result_score} != {expected_score}")
                    result_id = result.id
                    expected_id = expected_ids[j]
                    if result_id != expected_id:
                        print(f"GT id mismatch vector:{i} result:{j}/{search_limit}: {result_id} != {expected_id}")
        except KeyboardInterrupt:
            print("Bye - generating partial report")
            # break the loop and generate partial report
            break

    print("---------------------------------------")
    print("Search latency distribution:")
    quantiles = np.quantile(latency, [0, 0.5, 0.95, 0.99, 0.999, 1])
    print(f"min: {round(quantiles[0], 2)} millis")
    print(f"50p: {round(quantiles[1], 2)} millis")
    print(f"95p: {round(quantiles[2], 2)} millis")
    print(f"99p: {round(quantiles[3], 2)} millis")
    print(f"999p: {round(quantiles[4], 2)} millis")
    print(f"max: {round(quantiles[5], 2)} millis")
    print("")

    # query dimensions distribution
    if analyze_data:
        print("Query dimensions distribution:")
        quantiles = np.quantile(dimensions, [0, 0.5, 0.95, 0.99, 0.999, 1])
        print(f"min: {quantiles[0]}")
        print(f"50p: {quantiles[1]}")
        print(f"95p: {quantiles[2]}")
        print(f"99p: {quantiles[3]}")
        print(f"999p: {quantiles[4]}")
        print(f"max: {quantiles[5]}")

    # Create a 2D histogram of the query dimensions and latencies
    # https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
    timestamp = int(time.time())
    title = f"Sparse NeurIPS {dataset} ({segment_count} segments)"
    plt.hist2d(dimensions, latency, bins=100, cmap="rainbow")
    plt.grid(True)
    # force y-axis limits to be able to compare plots
    if graph_y_range is not None:
        axis = plt.gca()
        split = graph_y_range.split(" ")
        bottom = int(split[0])
        top = int(split[1])
        axis.set_ylim(bottom=bottom, top=top)
    cbar = plt.colorbar()
    cbar.set_label('Frequency')
    plt.xlabel('Query dimension count')
    plt.ylabel('Latency (ms)')
    plt.title(title)
    plot_file_name = f"./results/sparse_bench_{dataset}_{timestamp}.png"
    print(f"Saving plot to {plot_file_name}")
    plt.savefig(plot_file_name)
    plt.close()

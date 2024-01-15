
class DatasetStats:
    def __init__(self, posting_len_per_dimension, count_per_posting_len):
        self.posting_len_per_dimension = posting_len_per_dimension
        self.count_per_posting_len = count_per_posting_len


def compute_dataset_stats(csr_matrix) -> DatasetStats:
    print("Computing dataset stats...")
    vec_count = csr_matrix.shape[0]
    # compute posting length per dimension
    posting_len_per_dimension = {}
    for i in range(0, vec_count):
        point = csr_matrix[i]
        for index in point.indices:
            if index not in posting_len_per_dimension:
                posting_len_per_dimension[index] = 0
            posting_len_per_dimension[index] += 1
    # compute count per posting length
    count_per_posting_length = {}
    for index in posting_len_per_dimension:
        count = posting_len_per_dimension[index]
        if count not in count_per_posting_length:
            count_per_posting_length[count] = 0
        count_per_posting_length[count] += 1
    stats = DatasetStats(posting_len_per_dimension, count_per_posting_length)
    return stats

import logging
import _pickle as pickle

from datasketch import MinHashLSHForest, MinHash

import doppelspeller.settings as s
import doppelspeller.constants as c


LOGGER = logging.getLogger(__name__)


def generate_lsh_forest(ground_truth):
    LOGGER.info('Generating LSH forest for the ground truth data!')
    forest = MinHashLSHForest(num_perm=s.NUMBER_OF_PERMUTATIONS, l=32)
    number_of_rows_ground_truth = ground_truth.shape[0]

    for count, (title_id, sequences) in enumerate(zip(ground_truth.loc[:, c.COLUMN_TITLE_ID],
                                                      ground_truth.loc[:, c.COLUMN_SEQUENCES])):
        if not (count % 50000):
            LOGGER.info(f'Processed {count} out of {number_of_rows_ground_truth}!')

        min_hash = MinHash(num_perm=s.NUMBER_OF_PERMUTATIONS)
        _ = [min_hash.update(str(x).encode('utf8')) for x in sequences]
        forest.add(title_id, min_hash)

    LOGGER.info('Indexing the LSH forest!')
    forest.index()

    with open(s.LSH_FOREST_OUTPUT_FILE, 'wb') as file_object:
        pickle.dump(forest, file_object)

    return s.LSH_FOREST_OUTPUT_FILE

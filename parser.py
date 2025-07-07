from typing import Optional, Tuple
import numpy as np
import idx2numpy as idx2np

def parseIdx2Np(training_data_path: str,
                testing_data_path: str,
                training_labels_path: str,
                testing_labels_path: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parses IDX files into NumPy arrays.
    Args:
        training_data_path (str): Path to the training images IDX file.
        testing_data_path (str): Path to the testing images IDX file.
        training_labels_path (str): Path to the training labels IDX file.
        testing_labels_path (str): Path to the testing labels IDX file.

    Returns:
        Tuple[Optional[np.ndarray], ...]: A tuple containing
        (training_data, testing_data, training_labels, testing_labels),
        each element will be a NumPy array if all files are found,
        or None if a FileNotFoundError occurs.
    """
    
    try:
        # data -> NumPy arrays
        training_data = idx2np.convert_from_file(training_data_path).copy()
        testing_data = idx2np.convert_from_file(testing_data_path).copy()

        training_labels = idx2np.convert_from_file(training_labels_path).copy()
        testing_labels = idx2np.convert_from_file(testing_labels_path).copy()

        # visualize data shape
        # and format
        print ("Original NumPy shapes and Datatype:")
        print(training_data.shape)
        print(training_data.dtype)
        print(testing_data.shape)
        print(testing_data.dtype)
        print(training_labels.shape)
        print(training_labels.dtype)
        print(testing_labels.shape)
        print(testing_labels.dtype)

        print("-" * 30)

        return training_data, testing_data, training_labels, testing_labels

    except FileNotFoundError:
        raise ValueError("Warning: Data files not found.")

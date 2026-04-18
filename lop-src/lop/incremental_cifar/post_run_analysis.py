"""
Script for computing the effective rank, stable rank, number of dormant neurons, and average weight magnitude of the
models trained during the incremental cifar experiment.

Metric functions are imported from lop.metrics (canonical implementations).
"""

# built-in libraries
import time
import os
import argparse

# third party libraries
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from scipy.linalg import svd

# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, Normalize

from lop.nets.torchvision_modified_resnet import ResNet, build_resnet18

# Centralized metrics
from lop.metrics import (
    compute_average_weight_magnitude,
    compute_dormant_units_proportion,
    compute_effective_rank,
    compute_stable_rank,
)


# -------------------- For loading data and network parameters -------------------- #
def load_model_parameters(parameter_dir_path: str, index: int, epoch_number:int):
    """
    Loads the model parameters stored in parameter_dir_path corresponding to the index and epoch number
    return: torch module state dictionary
    """

    model_parameters_file_name = "index-{0}_epoch-{1}.pt".format(index, epoch_number)
    model_parameters_file_path = os.path.join(parameter_dir_path, model_parameters_file_name)

    if not os.path.isfile(model_parameters_file_path):
        error_message = "Couldn't find model parameters for index {0} and epoch number {1}.".format(index, epoch_number)
        raise ValueError(error_message)

    return torch.load(model_parameters_file_path)


def load_classes(classes_dir_path: str, index: int):
    """
    Loads the list of ordered classes used for partitioning the datta during the experiment
    return: list
    """

    classes_file_name = "index-{0}.npy".format(index)
    classes_file_path = os.path.join(classes_dir_path, classes_file_name)

    if not os.path.isfile(classes_file_path):
        error_message = "Couldn't find list of classes for index {0}.".format(index)
        raise ValueError(error_message)

    return np.load(classes_file_path)


def load_cifar_data(data_path: str, train: bool = True) -> (CifarDataSet, DataLoader):
    """
    Loads the cifar 100 data set with normalization
    :param data_path: path to the directory containing the data set
    :param train: bool that indicates whether to load the train or test data
    :return: torch DataLoader object
    """
    cifar_data = CifarDataSet(root_dir=data_path,
                              train=train,
                              cifar_type=100,
                              device=None,
                              image_normalization="max",
                              label_preprocessing="one-hot",
                              use_torch=True)

    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    transformations = [
        ToTensor(swap_color_axis=True),  # reshape to (C x H x W)
        Normalize(mean=mean, std=std),  # center by mean and divide by std
    ]

    cifar_data.set_transformation(transforms.Compose(transformations))

    num_workers = 12
    batch_size = 1000
    dataloader = DataLoader(cifar_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return cifar_data, dataloader


# NOTE: compute_average_weight_magnitude, compute_dormant_units_proportion,
# compute_effective_rank, and compute_stable_rank are now imported from
# lop.metrics (see imports above). The canonical implementations live there.



@torch.no_grad()
def compute_last_task_accuracy_per_class_in_order(net: torch.nn.Module, ordered_classes: np.ndarray,
                                                  test_data: DataLoader, experiment_index: int):
    """
    Computes the accuracy of each class in the order they were presented
    :param net: resnet with the parameters stored at the end of the experiment
    :param ordered_classes: numpy array with the cifar 100 classes in the order they were presented
    :param test_data: cifar100 test data
    :return: numpy array
    """

    ordered_classes = np.int32(ordered_classes)
    device = net.fc.weight.device
    num_classes = 100
    num_examples_per_class = 100

    class_correct = torch.zeros(num_classes, dtype=torch.float32, device=device)
    for i, sample in enumerate(test_data):
        image = sample["image"].to(device)
        labels = sample["label"].to(device)
        outputs = net(image)
        _, predicted = torch.max(outputs, 1)    # Get the class with the highest score
        _, labels = torch.max(labels, 1)        # Get the class with the highest score

        # Update the counts for each class
        for i, class_label in enumerate(ordered_classes):
            class_correct[i] += (predicted == labels).masked_select(labels == class_label).sum().item()

    return class_correct.cpu().numpy() / num_examples_per_class


# -------------------- For storing the results of the analysis -------------------- #
def store_analysis_results(weight_magnitude_results: np.ndarray,
                           dormant_units_results: (np.ndarray, np.ndarray),
                           effective_rank_results: (np.ndarray, np.ndarray),
                           stable_rank_results: (np.ndarray, np.ndarray),
                           accuracy_per_class_in_order: np.ndarray,
                           results_dir: str, experiment_index: int):
    """
    Stores the results of the post run analysis
    :param weight_magnitude_results: np array containing the output of compute_average_weight_magnitude
    :param dormant_units_results: tuple containing the results of the dormant unit analysis for the previous tasks and
                                  the next task for each different task
    :param effective_rank_results: tuple containing the results of the effective rank analysis for the previous tasks
                                   and the next task for each different task
    :param stable_rank_results: tuple containing the results of the stable rank analysis for the previous tasks and the
                                next task for each different task
    :param accuracy_per_class_in_order: np array containing the accuracy of the final model for each class in the order
                                        they were presented
    :param results_dir: path to the results directory
    :param experiment_index: experiment index
    """

    index_file_name = "index-{0}.npy".format(experiment_index)
    result_dir_names_and_arrays = [
        ("weight_magnitude_analysis", weight_magnitude_results),
        ("previous_tasks_dormant_units_analysis", dormant_units_results[0]),
        ("next_task_dormant_units_analysis", dormant_units_results[1]),
        ("previous_tasks_effective_rank_analysis", effective_rank_results[0]),
        ("next_task_effective_rank_analysis", effective_rank_results[1]),
        ("previous_tasks_stable_rank_analysis", stable_rank_results[0]),
        ("next_task_stable_rank_analysis", stable_rank_results[1]),
        ("accuracy_per_class_in_order", accuracy_per_class_in_order)
    ]

    # store results in the corresponding dir
    for results_name, results_array in result_dir_names_and_arrays:
        temp_results_dir = os.path.join(results_dir, results_name)
        os.makedirs(temp_results_dir, exist_ok=True)
        np.save(os.path.join(temp_results_dir, index_file_name), results_array)


def analyze_results(results_dir: str, data_path: str, dormant_unit_threshold: float = 0.01):
    """
    Analyses the parameters of a run and creates files with the results of the analysis
    :param results_dir: path to directory containing the results for a parameter combination
    :param data_path: path to the cifar100 data set
    :param dormant_unit_threshold: hidden units whose activation fall bellow this threshold are considered dormant
    """

    parameter_dir_path = os.path.join(results_dir, "model_parameters")
    experiment_indices_file_path = os.path.join(results_dir, "experiment_indices.npy")
    class_order_dir_path = os.path.join(results_dir, "class_order")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    number_of_epochs = np.arange(21) * 200  # by design the model parameters where store after each of these epochs)
    classes_per_task = 5                    # by design each task increases the data set by 5 classes
    last_epoch = 4000
    experiment_indices = np.load(experiment_indices_file_path)

    net = build_resnet18(num_classes=100, norm_layer=torch.nn.BatchNorm2d)
    net.to(device)
    cifar_data, cifar_data_loader = load_cifar_data(data_path, train=True)
    test_data, test_data_loader = load_cifar_data(data_path, train=False)

    for exp_index in tqdm(experiment_indices):
        ordered_classes = load_classes(class_order_dir_path, index=exp_index)

        average_weight_magnitude_per_epoch = np.zeros(number_of_epochs.size - 1, dtype=np.float32)
        dormant_units_prop_before = np.zeros_like(average_weight_magnitude_per_epoch)
        effective_rank_before = np.zeros_like(average_weight_magnitude_per_epoch)
        stable_rank_before = np.zeros_like(average_weight_magnitude_per_epoch)
        dormant_units_prop_after = np.zeros_like(average_weight_magnitude_per_epoch)
        effective_rank_after = np.zeros_like(average_weight_magnitude_per_epoch)
        stable_rank_after = np.zeros_like(average_weight_magnitude_per_epoch)

        for i, epoch_number in enumerate(number_of_epochs[:-1]):
            # get model parameters from before training on the task
            model_parameters = load_model_parameters(parameter_dir_path, index=exp_index, epoch_number=epoch_number)
            net.load_state_dict(model_parameters)

            # compute average weight magnitude
            average_weight_magnitude_per_epoch[i] = compute_average_weight_magnitude(net)

            # compute summaries for next task
            current_classes = ordered_classes[(i * classes_per_task):((i + 1) * classes_per_task)]
            cifar_data.select_new_partition(current_classes)

            prop_dormant, last_layer_features = compute_dormant_units_proportion(net, cifar_data_loader, dormant_unit_threshold)
            dormant_units_prop_after[i] = prop_dormant
            singular_values = svd(last_layer_features, compute_uv=False, lapack_driver="gesvd")
            effective_rank_after[i] = compute_effective_rank(singular_values)
            stable_rank_after[i] = compute_stable_rank(singular_values)

            # compute summaries from data from previous tasks
            if i == 0: continue
            current_classes = ordered_classes[:(i * classes_per_task)]
            cifar_data.select_new_partition(current_classes)
            prop_dormant, last_layer_features = compute_dormant_units_proportion(net, cifar_data_loader, dormant_unit_threshold)

            dormant_units_prop_before[i] = prop_dormant
            singular_values = svd(last_layer_features, compute_uv=False, lapack_driver="gesvd")
            effective_rank_before[i] = compute_effective_rank(singular_values)
            stable_rank_before[i] = compute_stable_rank(singular_values)

        net.load_state_dict(load_model_parameters(parameter_dir_path, exp_index, last_epoch))
        accuracy_per_class_in_order = compute_last_task_accuracy_per_class_in_order(net, ordered_classes,
                                                                                    test_data_loader, exp_index)

        store_analysis_results(weight_magnitude_results=average_weight_magnitude_per_epoch,
                               dormant_units_results=(dormant_units_prop_before, dormant_units_prop_after),
                               effective_rank_results=(effective_rank_before, effective_rank_after),
                               stable_rank_results=(stable_rank_before, stable_rank_after),
                               accuracy_per_class_in_order=accuracy_per_class_in_order,
                               results_dir=results_dir,
                               experiment_index=exp_index)


def parse_arguments() -> dict:

    file_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_dir', action="store", type=str,
                        default=os.path.join(file_path, "results", "base_deep_learning_system"),
                        help="Path to directory with the results of a parameter combination.")
    parser.add_argument('--data_path', action="store", type=str, default=os.path.join(file_path, "data"),
                        help="Path to directory with the CIFAR 100 data set.")
    parser.add_argument('--dormant_unit_threshold', action="store", type=float, default=0.01,
                        help="Units whose activations are less than this threshold are considered dormant.")

    args = parser.parse_args()
    return vars(args)


def main():

    analysis_arguments = parse_arguments()

    initial_time = time.perf_counter()
    analyze_results(results_dir=analysis_arguments["results_dir"],
                    data_path=analysis_arguments["data_path"],
                    dormant_unit_threshold=analysis_arguments["dormant_unit_threshold"])
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()

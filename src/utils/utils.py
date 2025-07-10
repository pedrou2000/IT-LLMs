def get_layer_node_indeces(node_idx: int, num_nodes_per_layer: int) -> tuple[int, int]:
    """
    Get the layer and node indices for a given node index in the model.

    :param model_info: ModelInformation object containing model details.
    :param node_idx: The node index to convert.
    :return: A tuple containing (layer_index, node_index).
    """
    layer_index = node_idx // num_nodes_per_layer
    node_index = node_idx % num_nodes_per_layer
    return layer_index, node_index

def get_node_index(layer_index: int, node_index: int, num_nodes_per_layer: int) -> int:
    """
    Get the node index for a given layer and node indices in the model.

    :param model_info: ModelInformation object containing model details.
    :param layer_index: The layer index.
    :param node_index: The node index within that layer.
    :return: The computed node index.
    """
    return layer_index * num_nodes_per_layer + node_index

def get_synapse_coordinates(Synapse,Neuron,Syn_prob,Range):
    """
    Calculates the 2D coordinates of each synapse based on a distance from the postsynaptic neuron.

    This function iterates through each synapse, calculates the vector from the presynaptic
    to the postsynaptic neuron, and then determines the synapse's position by moving
    a sampled distance along that vector from the postsynaptic neuron's position.

    No autaptic connections allowed

    Args:
        Synapse (brian2.Synapses): The Synapses object containing the connections.
        Neuron (brian2.NeuronGroup): The presynaptic/postsynaptic neuron group.


    Returns:
        list: A list of (x, y) tuples, where each tuple is the coordinate of a synapse.
    """
    # Initialize an empty list to store the synapse coordinates
    synapse_coords = []

    # Iterate through each synapse
    for i in range(len(synaptic_group)):
        # Get the indices of the pre- and postsynaptic neurons for the current synapse
        pre_idx = Synapse.i[i]
        post_idx = Synapse.j[i]

        # Get the coordinates of the pre- and postsynaptic neurons
        pre_pos = np.array([Neuron.x_neuron[pre_idx], Neuron.y_neuron[pre_idx]])
        post_pos = np.array([Neuron.x_neuron[post_idx], Neuron.y_neuron[post_idx]])

        # Calculate the vector from the postsynaptic to the presynaptic neuron
        vector = post_pos - pre_pos

        # Calculate the length (magnitude) of the vector
        vector_length = np.linalg.norm(vector)

        # Normalize the vector to get a unit vector
        unit_vector = vector / vector_length

        # Sample a distance for this synapse
        distance = sample_distance_from_soma(Syn_prob,Range)

        # Calculate the new coordinate by moving 'distance' along the unit vector from the postsynaptic neuron
        new_coord = post_pos + unit_vector * distance

        # Append the new coordinate as a tuple to the list
        synapse_coords.append(tuple(new_coord))

    return synapse_coords

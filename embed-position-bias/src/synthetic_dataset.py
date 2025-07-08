import numpy as np
import pandas as pd

def create_synthetic_dataset_multislot(
    seed=1, num_positions=5, num_items=5, step=30000,
    batch_size = 1000, item_proba = [0.1, 0.08, 0.06, 0.04, 0.02],
    position_proba = [1, 0.9, 0.4, 0.2, 0.1], assignment='fixed'
):
    """
    Generates a synthetic dataset representing a multi-slot environment.

    This function creates a dataset with items placed in various positions, each with an associated reward
    probability. The reward is calculated based on the item and position probabilities, which can represent
    the effectiveness of items in certain positions in a given context, such as advertisements on a webpage.

    Parameters:
    - seed (int): Random seed for reproducibility. Defaults to 1.
    - num_positions (int): The number of distinct positions available. Defaults to 5.
    - num_items (int): The number of different items that can be placed in positions. Defaults to 5.
    - step (int): The total number of data points to generate. Defaults to 30,000.
    - batch_size (int): Not currently used in the function, but can be included for future batching purposes. Defaults to 1000.
    - item_proba (list of float): A list of probabilities for the items being rewarded. Defaults to [0.1, 0.08, 0.06, 0.04, 0.02].
    - position_proba (list of float): A list of probabilities for rewards given the position of an item. Defaults to [1, 0.9, 0.4, 0.2, 0.1].
    - assignment (str): Determines how items are assigned to positions. Can be 'fixed' (each item gets its own position) or 'random' (positions are randomly assigned). Defaults to 'fixed'.

    Returns:
    - DataFrame: A pandas DataFrame with columns 'item', 'position', and 'reward', where 'reward' is a binary indicator (1 for reward, 0 for no reward).

    Raises:
    - ValueError: If the `assignment` parameter is not 'fixed' or 'random'.
    """    
    np.random.seed(seed)
    if item_proba is None:
        item_proba = np.random.random(num_items)
    if position_proba is None:
        position_proba = np.random.random(num_positions)
    items = np.random.randint(num_items, size=step)
    if assignment == 'fixed':
        positions = items
    elif assignment == 'random':
        positions = np.random.randint(position_proba, size=step)
    else:
        raise ValueError('arg assignment either "fixed" or "random".')
    df = pd.DataFrame({'item': items, 'position': positions})
    rewards = []
    for item, position in zip(items, positions):
        rewards.append(int(np.random.random() < item_proba[item] * position_proba[position]))
    df['reward'] = rewards
    return df


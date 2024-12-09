import os

def read_in(dir, label):
    """
    Reads in the reviews from given directory.

    Parameters
    ----------
    dir : str
        The directory to read the reviews from.
    
    label : str
        The label of the reviews set (1 or -1).

    Returns
    -------
    list[tuple[str, dict]]
        List of review tuples of its text and dict of its file id, rating, and label.
    """
    reviews = []
    for filename in os.listdir(dir):
        id, rating_str = filename.split("_")
        rating = int(rating_str.strip('.txt'))
        file_path = os.path.join(dir, filename)
        with open(file_path, 'r') as file:
            text = file.read()
        reviews.append((text, {"id": id, "rating": rating, "label": label}))
    return reviews


def get_lengths(dir):
    """
    Peace of mind.
    """
    lengths = []
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        with open(file_path, 'r') as file:
            text = file.read()
            lengths.append(len(text.split()))
            print(len(text))
    return lengths
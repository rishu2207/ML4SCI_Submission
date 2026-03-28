import random


def generate_dataset(n=50):
    prompts = [
        "Strong lensing with substructure",
        "Simple lensing image",
        "High resolution Einstein ring",
        "Noisy lensing simulation",
        "Clear arcs with distortion"
    ]

    return [random.choice(prompts) for _ in range(n)]


def split(data):
    split_idx = int(len(data) * 0.9)
    return data[:split_idx], data[split_idx:]
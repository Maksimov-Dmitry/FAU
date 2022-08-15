import pandas as pd

articles = pd.DataFrame(
    data={
        "AID": range(1, 16),
        "NAME": [
            "Apple",
            "Banana",
            "Kiwi",
            "Clementine",
            "Strawberry",
            "Cherry",
            "Carrot",
            "Bell Pepper",
            "Onion",
            "Salad",
            "Tomato",
            "Cucumber",
            "Spinach",
            "Water Melon",
            "Garlic",
        ],
        "PRICE": [
            0.3,
            0.75,
            0.6,
            0.5,
            0.45,
            0.4,
            0.5,
            0.7,
            0.3,
            0.4,
            0.45,
            0.3,
            0.6,
            1.5,
            0.3,
        ],
        "TYPE": [
            "Fruit",
            "Fruit",
            "Fruit",
            "Fruit",
            "Fruit",
            "Fruit",
            "Vegetable",
            "Vegetable",
            "Vegetable",
            "Vegetable",
            "Vegetable",
            "Vegetable",
            "Vegetable",
            "Fruit",
            "Vegetable",
        ],
    }
)

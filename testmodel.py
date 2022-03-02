from transformers import pipeline

model_id = "lewiswatson/distilbert-base-uncased-finetuned-emotion"
classifier = pipeline("text-classification", model=model_id)
labels = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]


def get_prediction(text):
    preds = classifier(text, return_all_scores=True)
    return display_predictions(preds)


# loop through predictions and display
def display_predictions(preds):
    count = 0
    for pred in preds[0]:
        print(f"{labels[count]} - {round(pred['score'], 3)}")
        count += 1
    return 1


def loop():
    while True:
        print()
        print()
        text = input("Enter a sentence: ")
        get_prediction(text)


loop()

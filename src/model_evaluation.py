from sklearn.metrics import classification_report

def evaluate_model(model, val_sequences, val_labels):
    """Evaluates the model and prints a classification report."""
    val_predictions = (model.predict(val_sequences) > 0.5).astype(int)
    print(classification_report(val_labels, val_predictions, target_names=['Non-toxic', 'Toxic']))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class Evaluator:
    @staticmethod
    def evaluate_model(model, X_test, y_test):

        y_pred = model.predict(X_test)
        if hasattr(y_pred, "shape") and len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
            y_pred = y_pred.argmax(axis=1)
        else:
            y_pred = (y_pred > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        }

import csv

# --- 1. DATA PREPARATION (BINNING) ---

def compute_bin_params(data, column, num_bins):
    """
    Scans `data` for valid numeric values in `column` and returns
    (min_val, bin_size, num_bins) so the same parameters can be
    reused on a separate test dataset.

    Returns None if no valid values are found.
    """
    valid_values = []
    for row in data:
        val = str(row.get(column, "")).strip()
        if val:
            try:
                valid_values.append(float(val))
            except ValueError:
                pass

    if not valid_values:
        return None

    min_val = min(valid_values)
    max_val = max(valid_values)

    # BUG FIX #1: guard against all-identical values (bin_size == 0)
    if max_val == min_val:
        bin_size = 1.0          # every value lands in the single bin [min, min+1)
    else:
        bin_size = (max_val - min_val) / num_bins

    return min_val, bin_size, num_bins


def apply_bins(data, column, bin_params):
    """
    Assigns a '<lower>-<upper>' group label to every row using the
    pre-computed bin parameters returned by compute_bin_params().

    Rows with a missing value get the label 'Unknown'.
    """
    if bin_params is None:
        # BUG FIX #2: if there were no valid values at all, mark every
        # row 'Unknown' so the key always exists (avoids KeyError later).
        for row in data:
            row[f"{column}Group"] = "Unknown"
        return

    min_val, bin_size, num_bins = bin_params

    for row in data:
        val = str(row.get(column, "")).strip()
        if not val:
            row[f"{column}Group"] = "Unknown"
        else:
            try:
                val = float(val)
            except ValueError:
                row[f"{column}Group"] = "Unknown"
                continue

            b_idx = int((val - min_val) / bin_size)
            if b_idx >= num_bins:
                b_idx = num_bins - 1      # clamp the maximum value
            if b_idx < 0:
                b_idx = 0                 # clamp values below training minimum

            lower = min_val + (b_idx * bin_size)
            upper = min_val + ((b_idx + 1) * bin_size)
            row[f"{column}Group"] = f"{int(lower)}-{int(upper)}"


# --- 2. ROW PREPROCESSING (shared between train and test) ---

def preprocess_row(row):
    """
    Converts spending columns to floats, computes total Spent,
    and simplifies the Cabin string to 'deck/side'.

    Mutates `row` in-place.
    """
    for column in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        val = str(row.get(column, "")).strip()
        row[column] = float(val) if val else 0.0

    row["Spent"] = (
        row["RoomService"]
        + row["FoodCourt"]
        + row["ShoppingMall"]
        + row["Spa"]
        + row["VRDeck"]
    )

    cabin = str(row.get("Cabin", "")).strip()
    if cabin:
        parts = cabin.split("/")
        if len(parts) == 3:
            # BUG FIX #3: was `f"{parts}/{parts}"` which stringified
            # the whole list object instead of indexing into it.
            row["Cabin"] = f"{parts[0]}/{parts[2]}"   # deck / side


# --- 3. NAIVE BAYES TRAINING ---

def train_naive_bayes(data, target_col, feature_cols):
    """
    Counts class frequencies and per-class feature value frequencies.

    Returns:
        class_counts   – {class_label: int}
        feature_counts – {class_label: {feature: {value: int}}}
        vocab          – {feature: set_of_all_seen_values}   (global vocab
                         used for consistent Laplace smoothing at predict time)
    """
    class_counts = {}
    feature_counts = {}
    vocab = {f: set() for f in feature_cols}   # BUG FIX #4 (see predict)

    for row in data:
        target = str(row.get(target_col, "")).strip()
        if not target:
            continue

        if target not in class_counts:
            class_counts[target] = 0
            feature_counts[target] = {f: {} for f in feature_cols}

        class_counts[target] += 1

        for f in feature_cols:
            val = str(row.get(f, "")).strip() or "Unknown"
            vocab[f].add(val)

            fc = feature_counts[target][f]
            fc[val] = fc.get(val, 0) + 1

    return class_counts, feature_counts, vocab


# --- 4. NAIVE BAYES PREDICTION ---

def predict_naive_bayes(row, class_counts, feature_counts, vocab, feature_cols):
    """
    Returns the most probable class label for `row`.

    Uses Laplace smoothing with the *global* vocabulary size so that
    the denominator is consistent across classes.

    BUG FIX #4: original code used per-class vocab size for the
    denominator, meaning two classes could have different smoothing
    denominators for the same feature — breaking the probability comparison.
    """
    total_rows = sum(class_counts.values())
    best_class = None
    highest_prob = -1.0

    for target_class, count in class_counts.items():
        prob = count / total_rows          # P(Class)

        for f in feature_cols:
            val = str(row.get(f, "")).strip() or "Unknown"

            f_count   = feature_counts[target_class][f].get(val, 0)
            # Use global vocab size so denominator is identical for every class
            vocab_size = len(vocab[f])

            # Laplace smoothing: (count + 1) / (class_total + vocab_size)
            prob *= (f_count + 1) / (count + vocab_size)

        if prob > highest_prob:
            highest_prob = prob
            best_class = target_class

    return best_class


# --- 5. PREDICT FROM TEST FILE AND WRITE SUBMISSION ---

def predict_and_save(test_file, output_file,
                     class_counts, feature_counts, vocab,
                     feature_cols, age_params, spent_params):
    """
    Reads `test_file` (CSV with no 'Transported' column),
    preprocesses each row the same way as training data,
    predicts the target, and writes a submission CSV to `output_file`.
    """
    test_rows = []
    with open(test_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            preprocess_row(row)
            test_rows.append(row)

    # Apply the SAME bin boundaries learned from the training data
    apply_bins(test_rows, "Age",   age_params)
    apply_bins(test_rows, "Spent", spent_params)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PassengerId", "Transported"])

        for row in test_rows:
            passenger_id = row.get("PassengerId", "Unknown").strip()
            prediction = predict_naive_bayes(
                row, class_counts, feature_counts, vocab, feature_cols
            )
            writer.writerow([passenger_id, prediction])

    print(f"\nPredictions saved to '{output_file}'.")


# --- MAIN EXECUTION ---

def main():
    # ------------------------------------------------------------------ #
    #  A. Load & preprocess training data                                  #
    # ------------------------------------------------------------------ #
    passenger_list = []

    with open("train.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            preprocess_row(row)
            passenger_list.append(row)

    # Compute bin boundaries from training data only
    age_params   = compute_bin_params(passenger_list, "Age",   num_bins=8)
    spent_params = compute_bin_params(passenger_list, "Spent", num_bins=6)

    # Apply bins to training data
    apply_bins(passenger_list, "Age",   age_params)
    apply_bins(passenger_list, "Spent", spent_params)

    # ------------------------------------------------------------------ #
    #  B. Train                                                            #
    # ------------------------------------------------------------------ #
    feature_cols = [
        "HomePlanet", "CryoSleep", "Cabin",
        "Destination", "VIP", "AgeGroup", "SpentGroup",
    ]
    target_col = "Transported"

    class_counts, feature_counts, vocab = train_naive_bayes(
        passenger_list, target_col, feature_cols
    )

    # ------------------------------------------------------------------ #
    #  C. Quick sanity-check on the first 10 training rows                #
    # ------------------------------------------------------------------ #
    print("Testing Naive Bayes Algorithm Predictions:")
    print("-" * 50)
    for i in range(min(10, len(passenger_list))):
        prediction = predict_naive_bayes(
            passenger_list[i], class_counts, feature_counts, vocab, feature_cols
        )
        actual = passenger_list[i].get(target_col, "?")
        name   = passenger_list[i].get("Name", f"Unknown Passenger {i}")
        print(f"{name:<24} | Predicted: {prediction:<5} | Actual: {actual}")

    # ------------------------------------------------------------------ #
    #  D. Predict on test file and write submission                        #
    # ------------------------------------------------------------------ #
    predict_and_save(
        test_file    = "test.csv",
        output_file  = "submission.csv",
        class_counts  = class_counts,
        feature_counts= feature_counts,
        vocab         = vocab,
        feature_cols  = feature_cols,
        age_params    = age_params,
        spent_params  = spent_params,
    )


if __name__ == "__main__":
    main()
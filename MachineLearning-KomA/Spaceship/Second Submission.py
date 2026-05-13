import csv
import math
from collections import Counter
from typing import List, Dict, Union, Optional, Any

class SpaceshipNaiveBayes:
    """
    A robust, object-oriented Naive Bayes Classifier tailored for the 
    Spaceship Titanic dataset. Implements log-probabilities for numerical 
    stability, percentile-based discretization for continuous variables, 
    and configurable Laplace smoothing.
    """
    
    def __init__(self, alpha: float = 1.0, num_bins_age: int = 8, num_bins_spent: int = 6):
        """
        Initializes the classifier.

        :param alpha: Laplace smoothing constant (default 1.0).
        :param num_bins_age: Number of percentile bins for the 'Age' feature.
        :param num_bins_spent: Number of percentile bins for the 'Spent' feature.
        """
        self.alpha = alpha
        self.num_bins_age = num_bins_age
        self.num_bins_spent = num_bins_spent
        
        self.class_counts = Counter()
        self.feature_counts = {}
        self.vocab = {}
        self.feature_cols = []
        self.target_col = ""
        self.total_rows = 0
        
        self.age_bin_edges = []
        self.spent_bin_edges = []
        
    def fit(self, data: List[Dict[str, str]], target_col: str = "Transported", feature_cols: List[str] = None):
        """
        Trains the Naive Bayes model on the provided dataset.
        """
        if feature_cols is None:
            self.feature_cols = ["HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP", "AgeGroup", "SpentGroup"]
        else:
            self.feature_cols = feature_cols
            
        self.target_col = target_col
        
        # Preprocess data (handles parsing and missing continuous values)
        processed_data = [self._preprocess_row(row) for row in data]
        
        # Compute bin edges using percentiles (handles outliers naturally)
        ages = [row["Age_Float"] for row in processed_data if row["Age_Float"] is not None]
        spents = [row["Spent"] for row in processed_data if row["Spent"] is not None]
        
        self.age_bin_edges = self._compute_percentile_edges(ages, self.num_bins_age)
        self.spent_bin_edges = self._compute_percentile_edges(spents, self.num_bins_spent)
        
        # Apply bins to create categorical features
        for row in processed_data:
            row["AgeGroup"] = self._apply_bin(row["Age_Float"], self.age_bin_edges)
            row["SpentGroup"] = self._apply_bin(row["Spent"], self.spent_bin_edges)
            
        # Compute frequencies using fast Counter objects
        self.class_counts = Counter()
        self.feature_counts = {}
        self.vocab = {f: set() for f in self.feature_cols}
        
        for row in processed_data:
            target = str(row.get(self.target_col, "")).strip()
            if not target:
                continue
                
            self.class_counts[target] += 1
            
            if target not in self.feature_counts:
                self.feature_counts[target] = {f: Counter() for f in self.feature_cols}
                
            for f in self.feature_cols:
                val = str(row.get(f, "")).strip()
                # Treat missing values as a distinct "Unknown" category
                if not val or val.lower() == "nan":
                    val = "Unknown"
                    
                self.vocab[f].add(val)
                self.feature_counts[target][f][val] += 1
                
        self.total_rows = sum(self.class_counts.values())
        
    def predict(self, data: Union[Dict[str, str], List[Dict[str, str]]]) -> Union[str, List[str]]:
        """
        Predicts the target class for a single dictionary or a list of dictionaries.
        """
        is_single = isinstance(data, dict)
        if is_single:
            data = [data]
            
        results = []
        for row in data:
            processed_row = self._preprocess_row(row)
            processed_row["AgeGroup"] = self._apply_bin(processed_row["Age_Float"], self.age_bin_edges)
            processed_row["SpentGroup"] = self._apply_bin(processed_row["Spent"], self.spent_bin_edges)
            
            class_log_probs = self._calculate_likelihood(processed_row)
            best_class = max(class_log_probs, key=class_log_probs.get) if class_log_probs else None
            results.append(best_class)
            
        return results[0] if is_single else results
        
    def _calculate_likelihood(self, row: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculates the log-likelihood of each class for a given row.
        Uses log-probabilities to prevent arithmetic underflow.
        """
        def calc_class_prob(target_class, count):
            log_prob = math.log(count / self.total_rows)
            for f in self.feature_cols:
                val = str(row.get(f, "")).strip()
                if not val or val.lower() == "nan":
                    val = "Unknown"
                    
                vocab_size = len(self.vocab[f])
                # If a test value was never seen during training, treat it as a new category 
                # to maintain mathematically sound smoothing.
                if val not in self.vocab[f]:
                    vocab_size += 1
                    
                f_count = self.feature_counts[target_class][f].get(val, 0)
                likelihood = (f_count + self.alpha) / (count + self.alpha * vocab_size)
                
                if likelihood > 0:
                    log_prob += math.log(likelihood)
                else:
                    # Fallback in case likelihood is somehow 0 (shouldn't happen with alpha > 0)
                    log_prob -= 1e9 
                    
            return log_prob
            
        # Dictionary comprehension for optimized and pythonic evaluation
        return {target_class: calc_class_prob(target_class, count) 
                for target_class, count in self.class_counts.items()}
        
    def _preprocess_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """
        Cleans a row: computes total spent, simplifies cabin structure, 
        and isolates numerical features for binning.
        """
        spent_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        total_spent = 0.0
        for col in spent_cols:
            val = str(row.get(col, "")).strip()
            if val and val.lower() != "nan":
                try:
                    total_spent += float(val)
                except ValueError:
                    pass
                    
        cabin = str(row.get("Cabin", "")).strip()
        if cabin and cabin.lower() != "nan":
            parts = cabin.split("/")
            if len(parts) == 3:
                cabin = f"{parts[0]}/{parts[2]}"  # deck / side
            else:
                cabin = "Unknown"
        else:
            cabin = "Unknown"
            
        age_val = str(row.get("Age", "")).strip()
        age_float = None
        if age_val and age_val.lower() != "nan":
            try:
                age_float = float(age_val)
            except ValueError:
                pass
                
        # Create a processed copy to avoid mutating original input
        processed = dict(row)
        processed["Spent"] = total_spent
        processed["Cabin"] = cabin
        processed["Age_Float"] = age_float
        
        return processed
        
    def _compute_percentile_edges(self, values: List[float], num_bins: int) -> List[float]:
        """
        Computes bin edges based on percentiles rather than min/max.
        Handles outliers inherently since extreme values just fall into the edge bins.
        """
        if not values:
            return []
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        edges = []
        
        for i in range(num_bins + 1):
            p = i / num_bins
            idx = p * (n - 1)
            lower_idx = math.floor(idx)
            upper_idx = math.ceil(idx)
            if lower_idx == upper_idx:
                val = sorted_vals[lower_idx]
            else:
                # Linear interpolation between indexes
                val = sorted_vals[lower_idx] + (idx - lower_idx) * (sorted_vals[upper_idx] - sorted_vals[lower_idx])
            edges.append(val)
            
        # Remove duplicate edges to avoid zero-width bins
        unique_edges = []
        for e in edges:
            if not unique_edges or abs(e - unique_edges[-1]) > 1e-9:
                unique_edges.append(e)
                
        # Guarantee at least 2 edges for 1 bin
        if len(unique_edges) < 2:
            unique_edges.append(unique_edges[-1] + 1.0)
            
        return unique_edges
        
    def _apply_bin(self, value: Optional[float], edges: List[float]) -> str:
        """
        Assigns a '<lower>-<upper>' group label to a value using 
        pre-computed percentile bin edges.
        """
        if value is None or not edges:
            return "Unknown"
            
        for i in range(len(edges) - 1):
            lower = edges[i]
            upper = edges[i+1]
            # Include the upper bound only for the last bin to catch max values
            if i == len(edges) - 2:
                if lower <= value <= upper:
                    return f"{int(lower)}-{int(upper)}"
            else:
                if lower <= value < upper:
                    return f"{int(lower)}-{int(upper)}"
                    
        # Fallback for test outliers outside of training percentiles
        if value < edges[0]:
            return f"<{int(edges[0])}"
        if value > edges[-1]:
            return f">{int(edges[-1])}"
            
        return "Unknown"


# --- MAIN EXECUTION ---

def main():
    # ------------------------------------------------------------------ #
    #  A. Load training data                                               #
    # ------------------------------------------------------------------ #
    passenger_list = []

    with open("train.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            passenger_list.append(row)

    # ------------------------------------------------------------------ #
    #  B. Train                                                            #
    # ------------------------------------------------------------------ #
    feature_cols = [
        "HomePlanet", "CryoSleep", "Cabin",
        "Destination", "VIP", "AgeGroup", "SpentGroup",
    ]
    
    model = SpaceshipNaiveBayes(alpha=1.0, num_bins_age=8, num_bins_spent=6)
    model.fit(passenger_list, target_col="Transported", feature_cols=feature_cols)

    # ------------------------------------------------------------------ #
    #  C. Quick sanity-check on the first 10 training rows                #
    # ------------------------------------------------------------------ #
    print("Testing Naive Bayes Algorithm Predictions:")
    print("-" * 50)
    
    test_subset = passenger_list[:10]
    predictions = model.predict(test_subset)
    
    for i, row in enumerate(test_subset):
        prediction = predictions[i]
        actual = row.get("Transported", "?")
        name = row.get("Name", f"Unknown Passenger {i}")
        print(f"{name:<24} | Predicted: {prediction:<5} | Actual: {actual}")

    # ------------------------------------------------------------------ #
    #  D. Predict on test file and write submission                        #
    # ------------------------------------------------------------------ #
    test_rows = []
    with open("test.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_rows.append(row)
            
    test_predictions = model.predict(test_rows)

    with open("submission.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PassengerId", "Transported"])

        for row, pred in zip(test_rows, test_predictions):
            passenger_id = row.get("PassengerId", "Unknown").strip()
            writer.writerow([passenger_id, pred])

    print(f"\nPredictions saved to 'submission.csv'.")

if __name__ == "__main__":
    main()
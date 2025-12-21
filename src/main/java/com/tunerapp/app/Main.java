package main.java.com.tunerapp.app;

import main.java.com.tunerapp.data.CrossValidator;
import main.java.com.tunerapp.tunablemodels.MyTunableRandomForest;
import main.java.com.tunerapp.tuner.GridSearchTuner;
import main.java.com.tunerapp.tuner.ParamSpace;
import main.java.com.tunerapp.tuner.TunableModel;
import main.java.com.tunerapp.data.CSVLoader;
import main.java.com.tunerapp.data.DataSplit;
import main.java.com.tunerapp.mlmodels.RandomForestSuite;
import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) {

        // Configuration
        String dataFilePath = "my_classification_data.csv";
        String targetColumn = "is_spam";
        boolean isClassificationTask = true;
        int random_state = 42;
        int k_folds = 5;

        try {
            // Data Loading and Splitting
            System.out.println("Loading data from: " + dataFilePath);
            CSVLoader.Dataset dataset = CSVLoader.load(dataFilePath, targetColumn, isClassificationTask);

            System.out.println("Splitting data...");
            DataSplit holdoutSplitter = new DataSplit(0.20, random_state);
            List<Object> splits = holdoutSplitter.train_test_split(dataset.X, dataset.y);

            double[][] X_train_for_tuning = (double[][]) splits.get(0);
            double[][] X_holdout_test = (double[][]) splits.get(1);
            int[] y_train_for_tuning = (int[]) splits.get(2);
            int[] y_holdout_test = (int[]) splits.get(3);

            System.out.printf("Data loaded: %d for tuning, %d for final holdout test.\n", X_train_for_tuning.length, X_holdout_test.length);

            // Tuner Configuration
            System.out.println("Configuring tuner for RandomForest...");
            TunableModel<double[][], int[]> myModel = new MyTunableRandomForest();

            ParamSpace space = new ParamSpace();
            space.add("n_estimators", Arrays.asList(50, 100, 150));
            space.add("max_depth", Arrays.asList(3, 5, 10));
            space.add("max_features", Arrays.asList(1, 2, 3));
            space.add("min_samples_split", Arrays.asList(2, 5, 10));

            // Cross-Validation Setup
            System.out.printf("Using %d-Fold Cross-Validation for evaluation.\n", k_folds);
            CrossValidator cv = new CrossValidator(X_train_for_tuning.length, k_folds, random_state);

            // Assembly and Execution
            System.out.println("Assembling and running the tuner...");
            GridSearchTuner<double[][], int[]> tuner = new GridSearchTuner<>(myModel, space);

            GridSearchTuner.TuningResult bestResult = tuner.tune(X_train_for_tuning, y_train_for_tuning, cv);

            if (bestResult == null || bestResult.bestParams == null) {
                System.err.println("Tuning failed to produce a valid result.");
                return;
            }

            // Display Results
            System.out.println("\n-------------------------------");
            System.out.println("    HYPERPARAMETER TUNING COMPLETE");
            System.out.println("---------------------------------");
            System.out.printf("Best Hyperparameters found: %s\n", bestResult.bestParams.toString());
            System.out.printf("Best Average CV Score (1.0 - F1_Score): %.4f\n", bestResult.bestScore);
            System.out.printf("Corresponding Best Average F1-Score: %.4f\n", (1.0 - bestResult.bestScore));

            // Final Model Evaluation
            System.out.println("\n--- Evaluating final model on the unseen holdout test set ---");

            RandomForestSuite.RandomForest finalModel = new RandomForestSuite.RandomForest(
                    (int) bestResult.bestParams.get("n_estimators"),
                    (int) bestResult.bestParams.get("max_depth"),
                    (int) bestResult.bestParams.get("min_samples_split"),
                    (int) bestResult.bestParams.get("max_features")
            );
            finalModel.fit(X_train_for_tuning, y_train_for_tuning);
            int[] y_final_pred = finalModel.predict(X_holdout_test);
            double final_f1_score = RandomForestSuite.ModelMetrics.f1_score(y_holdout_test, y_final_pred);

            System.out.printf("Final model F1-Score on holdout data: %.4f\n", final_f1_score);

        } catch (Exception e) {
            System.err.println("\n--- An unexpected error occurred! ---");
            e.printStackTrace();
        }
    }
}
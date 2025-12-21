package main.java.com.tunerapp.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class CrossValidator {

    private final int k;
    private final List<int[]> trainIndices;
    private final List<int[]> validationIndices;

    public static class Fold {
        public final int[] trainIndices;
        public final int[] validationIndices;
        public Fold(int[] train, int[] val) {
            this.trainIndices = train;
            this.validationIndices = val;
        }
    }

    public CrossValidator(int n_samples, int k, int random_state) {
        if (k < 2) {
            throw new IllegalArgumentException("Number of folds (k) must be at least 2.");
        }
        if (n_samples < k) {
            throw new IllegalArgumentException("Number of samples must be greater than or equal to k.");
        }
        this.k = k;
        this.trainIndices = new ArrayList<>();
        this.validationIndices = new ArrayList<>();

        // Create a shuffled list of indices from 0 to n_samples-1
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < n_samples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random(random_state));

        // Split the indices into k folds
        int foldSize = n_samples / k;
        List<List<Integer>> folds = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            int start = i * foldSize;
            // The last fold might be larger to account for remainders
            int end = (i == k - 1) ? n_samples : (i + 1) * foldSize;
            folds.add(new ArrayList<>(indices.subList(start, end)));
        }

        // For each fold, create the corresponding train and validation index arrays
        for (int i = 0; i < k; i++) {
            List<Integer> valFold = folds.get(i);
            List<Integer> trainFolds = new ArrayList<>();

            for (int j = 0; j < k; j++) {
                if (i != j) {
                    trainFolds.addAll(folds.get(j));
                }
            }

            // Store the final primitive arrays
            this.validationIndices.add(valFold.stream().mapToInt(Integer::intValue).toArray());
            this.trainIndices.add(trainFolds.stream().mapToInt(Integer::intValue).toArray());
        }
    }

    public int getK() {
        return this.k;
    }

    /**
     * Returns a specific train/validation fold.
     * @param foldIndex The index of the fold (from 0 to k-1).
     * @return A Fold object containing the train and validation indices.
     */
    public Fold getFold(int foldIndex) {
        if (foldIndex < 0 || foldIndex >= k) {
            throw new IndexOutOfBoundsException("Fold index must be between 0 and " + (k-1));
        }
        return new Fold(trainIndices.get(foldIndex), validationIndices.get(foldIndex));
    }
}
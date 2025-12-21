package main.java.com.tunerapp.tuner;

import main.java.com.tunerapp.data.CrossValidator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

/**
 * A Grid Search hyperparameter tuner that utilizes all available CPU cores
 * by running model training trials in parallel using a Java thread pool.
 * This version uses K-Fold Cross-Validation for robust model evaluation.
 *
 * @param <X_TYPE> The generic type for feature data.
 * @param <Y_TYPE> The generic type for target data.
 */
public class GridSearchTuner<X_TYPE, Y_TYPE> {

    private final TunableModel<X_TYPE, Y_TYPE> modelTemplate;
    private final ParamSpace paramSpace;

    /**
     * Holds the final result of the entire tuning process. Declared as 'static'
     * so it can be easily referenced from other classes like Main.
     */
    public static class TuningResult {
        public final Map<String, Object> bestParams;
        public final double bestScore;
        public final List<TrialResult> allTrialResults;

        public TuningResult(Map<String, Object> bestParams, double bestScore, List<TrialResult> allTrials) {
            this.bestParams = bestParams;
            this.bestScore = bestScore;
            this.allTrialResults = allTrials;
        }

        @Override
        public String toString() {
            return "TuningResult{" + "bestParams=" + bestParams + ", bestScore=" + bestScore + '}';
        }
    }

    /**
     * Holds the result of a single model training trial. Declared as 'static'
     * for clean, independent usage.
     */
    public static class TrialResult {
        public final Map<String, Object> params;
        public final double score;

        public TrialResult(Map<String, Object> params, double score) {
            this.params = params;
            this.score = score;
        }
    }

    public GridSearchTuner(TunableModel<X_TYPE, Y_TYPE> model, ParamSpace paramSpace) {
        this.modelTemplate = model;
        this.paramSpace = paramSpace;
    }

    public TuningResult tune(X_TYPE full_X_train, Y_TYPE full_y_train, CrossValidator cv) {
        List<Map<String, Object>> allCombinations = generateCombinations();
        System.out.printf("Generated %d total combinations to test.\n", allCombinations.size());

        int numCores = Runtime.getRuntime().availableProcessors();
        System.out.printf("Starting parallel grid search on %d CPU cores using %d-fold CV.\n", numCores, cv.getK());
        ExecutorService executor = Executors.newFixedThreadPool(numCores);

        List<Future<TrialResult>> futures = new ArrayList<>();

        for (Map<String, Object> params : allCombinations) {
            Callable<TrialResult> task = () -> {
                modelTemplate.setParams(params);
                double score = modelTemplate.fitAndEvaluate(full_X_train, full_y_train, cv);

                if (Double.isNaN(score) || Double.isInfinite(score)) {
                    score = Double.MAX_VALUE; // Assign max penalty for failed trials
                }
                System.out.printf("  -> Finished trial with params: %s, Avg CV Score: %.4f\n", params, score);
                return new TrialResult(params, score);
            };
            futures.add(executor.submit(task));
        }

        executor.shutdown();
        try {
            if (!executor.awaitTermination(2, TimeUnit.HOURS)) {
                System.err.println("Grid search timed out and did not complete.");
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            System.err.println("Grid search was interrupted.");
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        List<TrialResult> allResults = new ArrayList<>();
        TrialResult bestResult = null;
        for (Future<TrialResult> future : futures) {
            try {

                if (future.isDone()) {
                    TrialResult currentResult = future.get();
                    allResults.add(currentResult);

                    if (bestResult == null || currentResult.score < bestResult.score) {
                        bestResult = currentResult;
                    }
                }
            } catch (InterruptedException | ExecutionException e) {
                System.err.println("A trial failed with an exception: " + e.getMessage());
            }
        }

        if (bestResult == null) {
            throw new RuntimeException("Grid Search completed without any successful trials.");
        }

        return new TuningResult(bestResult.params, bestResult.score, allResults);
    }

    private List<Map<String, Object>> generateCombinations() {
        List<Map<String, Object>> combinations = new ArrayList<>();
        List<String> paramNames = paramSpace.getParamNames();
        if (paramNames.isEmpty()) {
            return combinations;
        }

        generateRecursive(0, new HashMap<>(), paramNames, combinations);
        return combinations;
    }

    private void generateRecursive(int paramIndex, Map<String, Object> currentCombination,
                                   List<String> paramNames, List<Map<String, Object>> finalCombinations) {
        if (paramIndex == paramNames.size()) {
            finalCombinations.add(new HashMap<>(currentCombination));
            return;
        }

        String currentParamName = paramNames.get(paramIndex);
        List<?> values = paramSpace.getValues(currentParamName);

        for (Object value : values) {
            currentCombination.put(currentParamName, value);
            generateRecursive(paramIndex + 1, currentCombination, paramNames, finalCombinations);
        }
    }
}
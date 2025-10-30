package javaml;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import static javaml.MakeRegression.makeRegression;

public class JavaHyperParameterTuner
{
    public static void main(String[] args) throws ExecutionException, InterruptedException {
        // --- 0. Create dummy data ---
        RegressionData data = makeRegression(1000000, 5, 30, 13);
        DataSplit ds = new DataSplit(0.2,3);
        List<Object> splits = ds.train_test_split(data.X, data.y);

        double[][] X_train = (double[][]) splits.get(0);
        double[][] X_test = (double[][]) splits.get(1);
        double[] y_train = (double[]) splits.get(2);
        double[] y_test = (double[]) splits.get(3);

        // --- 1. Define the hyperparameter search space ---
        int[] epochOptions = {50, 100, 200,300,250};
        double[] lrOptions = {0.1, 0.01, 0.001,0.0001,0.00001};
        int numParallelJobs = 4;

        // --- 2. Create the list of all tasks to be run ---
        List<Callable<JobResult>> tasks = new ArrayList<>();
        for (int epochs : epochOptions) {
            for (double lr : lrOptions) {
                // The task is the same, we just add it to a list instead of submitting it directly.
                tasks.add(new TrainingTask(epochs, lr, X_train, y_train, X_test, y_test));
            }
        }

        // --- 3. Use the ParallelJobRunner to execute the tasks ---
        ParallelJobRunner<JobResult> jobRunner = new ParallelJobRunner<>(numParallelJobs);
        long startTime = System.currentTimeMillis();

        // This single line runs all jobs in parallel and waits for them to finish.
        List<Future<JobResult>> allFutures = jobRunner.runJobs(tasks);

        System.out.println("All jobs completed processing.");
        long endTime = System.currentTimeMillis();

        // --- 4. Process the results (same as before) ---
        JobResult bestResult = null;
        for (Future<JobResult> future : allFutures) {
            JobResult currentResult = future.get();
            if (bestResult == null || currentResult.getScore() < bestResult.getScore()) {
                bestResult = currentResult;
            }
        }

        // --- 5. Report the best result and shut down the runner ---
        System.out.println("\n----------------- Tuning Complete -----------------");
        if (bestResult != null) {
            System.out.println("Best (Lowest) MSE: " + String.format("%.4f", bestResult.getScore()));
            System.out.println("Found with Hyperparameters: " + bestResult.getParams());
        }
        System.out.println("Total time taken: " + (endTime - startTime) + " ms");

        // Gracefully shut down the thread pool inside the runner.
        jobRunner.shutdown();
    }
}

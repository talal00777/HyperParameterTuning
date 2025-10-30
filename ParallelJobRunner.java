package javaml;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

public class ParallelJobRunner<T> {

    private final ExecutorService executor;

    /**
     * Creates a job runner with a fixed number of threads.
     * @param numThreads The number of jobs to run in parallel.
     */
    public ParallelJobRunner(int numThreads) {
        if (numThreads <= 0) {
            throw new IllegalArgumentException("Number of threads must be positive.");
        }
        this.executor = Executors.newFixedThreadPool(numThreads);
    }

    /**
     * Executes a list of tasks concurrently and waits for all of them to complete.
     *
     * @param tasks The list of Callable tasks to execute.
     * @return A list of Future objects, each holding the eventual result of a task.
     */
    public List<Future<T>> runJobs(List<Callable<T>> tasks) {
        System.out.println("Submitting " + tasks.size() + " jobs to the thread pool...");

        List<Future<T>> futures = new ArrayList<>();
        try {
            // invokeAll executes all tasks and waits for them all to complete.
            futures = executor.invokeAll(tasks);
        } catch (InterruptedException e) {
            System.err.println("Thread pool was interrupted while waiting for tasks to complete.");
            Thread.currentThread().interrupt();
        }

        System.out.println("All jobs have been submitted and are being processed...");
        return futures;
    }

    /**
     * Shuts down the thread pool gracefully. This should be called when
     * the application is finished with the runner.
     */
    public void shutdown() {
        System.out.println("Shutting down the thread pool.");
        executor.shutdown();
        try {
            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}

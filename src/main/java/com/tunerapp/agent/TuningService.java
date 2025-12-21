// File: TuningService.java (Corrected with isClassification parameter)
package main.java.com.tunerapp.agent;

import main.java.com.tunerapp.data.CSVLoader;
import main.java.com.tunerapp.data.CrossValidator;
import main.java.com.tunerapp.data.DataSplit;
import main.java.com.tunerapp.tunablemodels.MyTunableMGDRegressor;
import main.java.com.tunerapp.tunablemodels.MyTunableRandomForest;
import main.java.com.tunerapp.tuner.GridSearchTuner;
import main.java.com.tunerapp.tuner.ParamSpace;
import main.java.com.tunerapp.tuner.TunableModel;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.*;

public class TuningService {

    private final Map<String, JobStatus> jobStatuses = new ConcurrentHashMap<>();
    private final ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

    // --- THIS IS THE CORRECTED METHOD SIGNATURE ---
    public String startNewTuningJob(InputStream dataStream, String modelName, ParamSpace space, boolean isClassification) {
        String jobId = UUID.randomUUID().toString();
        jobStatuses.put(jobId, new JobStatus(jobId, "QUEUED", "Job is waiting in the queue."));

        Runnable tuningTask = () -> {
            JobStatus currentJobStatus = jobStatuses.get(jobId);
            try {
                currentJobStatus.status = "PREPARING_DATA";
                currentJobStatus.message = "Loading and splitting data...";

                // --- NOW THIS USES THE isClassification PARAMETER ---
                CSVLoader.Dataset dataset = CSVLoader.load(new InputStreamReader(dataStream), "target", isClassification);

                DataSplit holdoutSplitter = new DataSplit(0.20, 42);
                List<Object> splits = holdoutSplitter.train_test_split(dataset.X, dataset.y);

                double[][] X_train_for_tuning = (double[][]) splits.get(0);
                Object y_train_for_tuning_obj = splits.get(2); // Keep as Object for now
                int n_samples = X_train_for_tuning.length;

                currentJobStatus.status = "RUNNING";
                currentJobStatus.message = "Tuning process has started. This may take a while...";

                // --- NOW THIS LOGIC USES THE isClassification PARAMETER ---
                TunableModel model;
                if (isClassification) {
                    model = new MyTunableRandomForest();
                } else {
                    model = new MyTunableMGDRegressor();
                }

                int k_folds = 5;
                CrossValidator cv = new CrossValidator(n_samples, k_folds, 42);

                // We can now safely create the tuner with the right model
                GridSearchTuner tuner = new GridSearchTuner<>(model, space);
                GridSearchTuner.TuningResult result = tuner.tune(X_train_for_tuning, y_train_for_tuning_obj, cv);

                // --- COMPLETION (No change here) ---
                currentJobStatus.status = "COMPLETED";
                currentJobStatus.message = "Tuning finished successfully!";
                currentJobStatus.bestParams = result.bestParams;
                currentJobStatus.bestScore = result.bestScore;

            } catch (Exception e) {
                currentJobStatus.status = "FAILED";
                currentJobStatus.message = "An error occurred: " + e.getMessage();
                e.printStackTrace();
            }
        };

        executor.submit(tuningTask);
        return jobId;
    }

    public JobStatus getJobStatus(String jobId) {
        return jobStatuses.get(jobId);
    }
}
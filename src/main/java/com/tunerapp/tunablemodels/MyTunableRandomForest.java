package main.java.com.tunerapp.tunablemodels;

import main.java.com.tunerapp.data.CrossValidator;
import main.java.com.tunerapp.tuner.TunableModel;
import main.java.com.tunerapp.mlmodels.RandomForestSuite;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.lang.Math;

public class MyTunableRandomForest implements TunableModel<double[][], int[]> {

    private int nEstimators;
    private int maxDepth;
    private int minSamplesSplit;
    private int maxFeatures;
    private int nFeaturesInDataset = -1;

    @Override
    public void setParams(Map<String, Object> params) {

        this.nEstimators = (int) params.getOrDefault("n_estimators", 100);
        this.maxDepth = (int) params.getOrDefault("max_depth", 10);
        this.minSamplesSplit = (int) params.getOrDefault("min_samples_split", 2);
        this.maxFeatures = (int) params.getOrDefault("max_features", -1);
    }

    @Override
    public double fitAndEvaluate(double[][] full_X_train, int[] full_y_train, CrossValidator cv) {

        List<Double> foldScores = new ArrayList<>();

        if (this.nFeaturesInDataset == -1 && full_X_train.length > 0) {
            this.nFeaturesInDataset = full_X_train[0].length;
        }

        for (int i = 0; i < cv.getK(); i++)
        {
            CrossValidator.Fold fold = cv.getFold(i);
            double[][] X_fold_train = getSubset(full_X_train, fold.trainIndices);
            int[] y_fold_train = getSubset(full_y_train, fold.trainIndices);
            double[][] X_fold_val = getSubset(full_X_train, fold.validationIndices);
            int[] y_fold_val = getSubset(full_y_train, fold.validationIndices);

            int actualMaxFeatures = this.maxFeatures;
            if (actualMaxFeatures == -1 && this.nFeaturesInDataset > 0) {
                actualMaxFeatures = (int) Math.round(Math.sqrt(this.nFeaturesInDataset));
            } else if (actualMaxFeatures == -1) {
                actualMaxFeatures = 1; // Fallback
            }

            RandomForestSuite.RandomForest rf = new RandomForestSuite.RandomForest(
                    this.nEstimators,
                    this.maxDepth,
                    this.minSamplesSplit,
                    actualMaxFeatures
            );

            rf.fit(X_fold_train, y_fold_train);
            int[] y_pred = rf.predict(X_fold_val);
            double f1_score = RandomForestSuite.ModelMetrics.f1_score(y_fold_val, y_pred);
            foldScores.add(1.0 - f1_score);
        }

        return foldScores.stream().mapToDouble(d -> d).average().orElse(Double.MAX_VALUE);
    }

    private double[][] getSubset(double[][] original, int[] indices) {
        double[][] subset = new double[indices.length][];
        for (int i = 0; i < indices.length; i++) {
            subset[i] = original[indices[i]];
        }
        return subset;
    }

    private int[] getSubset(int[] original, int[] indices) {
        int[] subset = new int[indices.length];
        for (int i = 0; i < indices.length; i++) {
            subset[i] = original[indices[i]];
        }
        return subset;
    }
}
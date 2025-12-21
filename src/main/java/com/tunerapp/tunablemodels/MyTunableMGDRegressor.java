package main.java.com.tunerapp.tunablemodels;

import main.java.com.tunerapp.data.CrossValidator;
import main.java.com.tunerapp.tuner.TunableModel;
import main.java.com.tunerapp.mlmodels.MGDRegressor;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class MyTunableMGDRegressor implements TunableModel<double[][], double[]> {

    private int epochs;
    private double lr;

    @Override
    public void setParams(Map<String, Object> params)
    {
        this.epochs = (int) params.getOrDefault("epochs", 100);
        this.lr = (double) params.getOrDefault("lr", 0.01);
    }

    @Override
    public double fitAndEvaluate(double[][] full_X_train, double[] full_y_train, CrossValidator cv) {

        List<Double> foldScores = new ArrayList<>();

        // --- THE CROSS-VALIDATION LOOP ---
        for (int i = 0; i < cv.getK(); i++) {
            CrossValidator.Fold fold = cv.getFold(i);

            // Get the specific train/validation data for this fold
            double[][] X_fold_train = getSubset(full_X_train, fold.trainIndices);
            double[] y_fold_train = getSubset(full_y_train, fold.trainIndices);
            double[][] X_fold_val = getSubset(full_X_train, fold.validationIndices);
            double[] y_fold_val = getSubset(full_y_train, fold.validationIndices);

            // Instantiate and train a new model for this fold
            MGDRegressor regressor = new MGDRegressor(this.epochs, this.lr);
            regressor.train(X_fold_train, y_fold_train);

            // Evaluate and store the score for this fold
            double[] y_pred = regressor.predict(X_fold_val);
            // Use a stable metric like MAE for evaluation. The tuner MINIMIZES this score.
            double score = MGDRegressor.mae(y_fold_val, y_pred);
            foldScores.add(score);
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

    private double[] getSubset(double[] original, int[] indices) {
        double[] subset = new double[indices.length];
        for (int i = 0; i < indices.length; i++) {
            subset[i] = original[indices[i]];
        }
        return subset;
    }
}
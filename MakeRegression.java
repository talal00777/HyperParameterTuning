package javaml;

import java.util.Random;

public class MakeRegression {
    public static RegressionData makeRegression(int nSamples, int nFeatures, double noise, int seed) {
        Random rand = new Random(seed);
        double[][] X = new double[nSamples][nFeatures];
        double[] weights = new double[nFeatures];
        double[] y = new double[nSamples];

        for (int j = 0; j < nFeatures; j++) {
            weights[j] = rand.nextDouble() * 10 - 5;
        }

        for (int i = 0; i < nSamples; i++) {
            double sum = 0;
            for (int j = 0; j < nFeatures; j++) {
                X[i][j] = rand.nextDouble() * 10;
                sum += X[i][j] * weights[j];
            }
            y[i] = sum + noise * rand.nextGaussian();
        }

        return new RegressionData(X, y);
    }
}

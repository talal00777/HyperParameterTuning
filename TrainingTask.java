package javaml;

import java.util.concurrent.Callable;

class TrainingTask implements Callable<JobResult> {
    private final int epochs;
    private final double learningRate;
    private final double[][] X_train, X_test;
    private final double[] y_train, y_test;

    public TrainingTask(int epochs, double learningRate, double[][] X_train, double[] y_train, double[][] X_test, double[] y_test) {
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.X_train = X_train;
        this.y_train = y_train;
        this.X_test = X_test;
        this.y_test = y_test;
    }

    @Override
    public JobResult call() {
        System.out.println(Thread.currentThread().getName() + ": Starting job with " + "LR=" + learningRate + ", Epochs=" + epochs);
        MGDRegressor mgd = new MGDRegressor(epochs,learningRate);
        mgd.train(X_train, y_train);

        double[] y_pred = mgd.predict(X_test);
        // This now calls the method that calculates real MSE
        int num_features = X_test[0].length;
        double adj_r2 = MGDRegressor.adjusted_r2(y_test, y_pred,num_features);

        System.out.println(Thread.currentThread().getName() + ": Finished job with " + "LR=" + learningRate + ", ADJ_R2 = " + String.format("%.4f", adj_r2));
        return new JobResult(adj_r2, mgd.getParams());
    }
}
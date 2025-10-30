package javaml;

import java.util.*;

public class DataSplit {
    double test_size;
    int random_state;

    public DataSplit(double test_size, int random_state) {
        this.test_size = test_size;
        this.random_state = random_state;
    }

    public List<Object> train_test_split(double[][] X, double[] y) {
        Random random = new Random(this.random_state);
        int n = X.length;

        // --- Step 1: Create an array of indices 0, 1, 2, ..., n-1
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }

        // --- Step 2: Shuffle the indices randomly
        List<Integer> indicesList = Arrays.asList(indices);
        Collections.shuffle(indicesList, random);
        indicesList.toArray(indices);

        // --- Step 3: Compute number of test samples
        int num_test_samples = (int) Math.round(test_size * n);

        // --- Step 4: Split into test and train indices
        Integer[] test_indices = Arrays.copyOfRange(indices, 0, num_test_samples);
        Integer[] train_indices = Arrays.copyOfRange(indices, num_test_samples, n);

        // --- Step 5: Create lists for split data
        List<double[]> X_train = new ArrayList<>();
        List<Double> y_train = new ArrayList<>();
        List<double[]> X_test = new ArrayList<>();
        List<Double> y_test = new ArrayList<>();

        // --- Step 6: Fill training data
        for (int idx : train_indices) {
            X_train.add(X[idx]);
            y_train.add(y[idx]);
        }

        // --- Step 7: Fill test data
        for (int idx : test_indices) {
            X_test.add(X[idx]);
            y_test.add(y[idx]);
        }

        // --- Step 8: Return results as a list of objects (similar to Python tuple)
        double[][] X_train_arr = X_train.toArray(new double[X_train.size()][]);
        double[][] X_test_arr  = X_test.toArray(new double[X_test.size()][]);
        double[] y_train_arr   = y_train.stream().mapToDouble(Double::doubleValue).toArray();
        double[] y_test_arr    = y_test.stream().mapToDouble(Double::doubleValue).toArray();

        return List.of(X_train_arr, X_test_arr, y_train_arr, y_test_arr);

    }

}



/*public class DataSplit
{
    double test_size;
    int random_state;
    double[][] X;
    double[] y;

    public DataSplit(double test_size, int random_state)
    {
        this.test_size = test_size;
        this.random_state = random_state;
    }

    public List<double[]>  train_test_split(double[][] X, double[] y)
    {
        Random random = new Random(this.random_state);

        Integer[] indices = new Integer[0];

        for(Integer i = 0; i<X.length;i++)
        {
            indices[i] = i;
        }

        List<Integer> indicess = Arrays.asList(indices);

        Collections.shuffle(indicess);

        Integer[] indexes = indicess.toArray(new Integer[0]);

        int num_test_samples = X.length * Integer.parseInt(String.valueOf(this.test_size));

        Integer[] test_indices = new Integer[0];

        for(Integer i = 0; i < num_test_samples; i++)
        {
            test_indices[i] = indexes[i];
        }

        Integer[] train_indices = new Integer[0];

        for(int i = num_test_samples; i < indexes.length; i++)
        {
            train_indices[i] = indexes[i];
        }


        List<double[]> X_train = new ArrayList<>();
        List<Double> y_train = new ArrayList<>();
        List<double[]> X_test = new ArrayList<>();
        List<Double> y_test = new ArrayList<>();

        for (int i : train_indices)
        {
            X_train.add(X[i]);
            y_train.add(y[i]);
        }

        for (int i : test_indices)
        {
            X_test.add(X[i]);
            y_test.add(y[i]);
        }

        return [X_test,X_train,y_test,y_train];
    }
}*/

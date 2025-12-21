package main.java.com.tunerapp.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class DataSplit
{
    double test_size;
    int random_state;

    public DataSplit(double test_size, int random_state)
    {
        this.test_size = test_size;
        this.random_state = random_state;
    }

    public List<Object> train_test_split(double[][] X, Object y)
    {
        int n_samples = X.length;
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < n_samples; i++) indices.add(i);

        Collections.shuffle(indices, new Random(this.random_state));

        int test_count = (int) (n_samples * this.test_size);
        int train_count = n_samples - test_count;

        List<Integer> train_indices = indices.subList(0, train_count);
        List<Integer> test_indices = indices.subList(train_count, n_samples);

        double[][] X_train = new double[train_count][];
        double[][] X_test = new double[test_count][];
        for (int i = 0; i < train_count; i++) X_train[i] = X[train_indices.get(i)];
        for (int i = 0; i < test_count; i++) X_test[i] = X[test_indices.get(i)];

        Object y_train, y_test;

        if (y instanceof int[])
        {
            int[] y_vector = (int[]) y;
            int[] y_train_vector = new int[train_count];
            int[] y_test_vector = new int[test_count];
            for (int i = 0; i < train_count; i++) y_train_vector[i] = y_vector[train_indices.get(i)];
            for (int i = 0; i < test_count; i++) y_test_vector[i] = y_vector[test_indices.get(i)];
            y_train = y_train_vector;
            y_test = y_test_vector;

        }
        else if (y instanceof double[])
        {
            double[] y_vector = (double[]) y;
            double[] y_train_vector = new double[train_count];
            double[] y_test_vector = new double[test_count];
            for (int i = 0; i < train_count; i++) y_train_vector[i] = y_vector[train_indices.get(i)];
            for (int i = 0; i < test_count; i++) y_test_vector[i] = y_vector[test_indices.get(i)];
            y_train = y_train_vector;
            y_test = y_test_vector;

        }
        else if (y instanceof double[][])
        {
            double[][] y_matrix = (double[][]) y;
            double[][] y_train_matrix = new double[train_count][];
            double[][] y_test_matrix = new double[test_count][];
            for (int i = 0; i < train_count; i++) y_train_matrix[i] = y_matrix[train_indices.get(i)];
            for (int i = 0; i < test_count; i++) y_test_matrix[i] = y_matrix[test_indices.get(i)];
            y_train = y_train_matrix;
            y_test = y_test_matrix;

        }
        else
        {
            throw new IllegalArgumentException("Unsupported type for y. Must be double[][], double[], or int[].");
        }

        List<Object> result = new ArrayList<>();
        result.add(X_train);
        result.add(X_test);
        result.add(y_train);
        result.add(y_test);

        return result;
    }
}
package javaml;

import java.util.Arrays;
import static java.lang.Math.pow;

public class MGDRegressor
{
    int epochs;
    double learning_rate;
    double[] weights;
    Double intercept;

    public MGDRegressor(int epochs, double learning_rate)
    {
        this.epochs = epochs;
        this.learning_rate = learning_rate;
        this.weights = null;
        this.intercept = null;
    }

    public void train(double[][] X_train, double[] y_train )
    {
        int num_features = X_train[0].length;
        int num_data_points = X_train.length;

        this.weights = new double[num_features];
        this.intercept = 0.0;

        for (int i = 0; i < this.epochs; i++)
        {
            double[] weights_gradient = new double[num_features];
            double intercept_gradient = 0.0;

            for(int j = 0; j < num_data_points; j++)
            {
                double[] data_point = X_train[j];
                double target = y_train[j];
                double y_pred = 0.0;

                for (int k = 0; k < num_features; k++)
                {
                    y_pred += this.weights[k] * data_point[k];
                }

                y_pred += this.intercept;
                double error = y_pred - target;
                intercept_gradient += error;

                for(int k = 0; k < num_features; k++)
                {
                    weights_gradient[k] += error * data_point[k];
                }

                this.intercept = this.intercept - (this.learning_rate * ((double) 2 /num_data_points) * intercept_gradient);

                for (int l = 0; l < num_features; l++)
                {
                    this.weights[l] = this.weights[l] - (this.learning_rate * ((double) 2 /num_data_points) * weights_gradient[l]);
                }

            }
        }
    }

    public double[] predict(double[][] X_test)
    {
        double[] predictions = new double[X_test.length];
        int num_test_samples = X_test.length;
        int num_features = X_test[0].length;
        predictions = Arrays.copyOf(predictions, predictions.length + 1);


        for (int i = 0; i < num_test_samples; i++)
        {
            double[] data_point = X_test[i];
            double prediction = 0.0;

            for (int j = 0; j < num_features; j++)
            {
                prediction += this.weights[j] * data_point[j];
            }

            prediction += this.intercept;

            predictions[i] = prediction;
        }

        return predictions;

    }

    public static double mse(double[] y_test,double[] y_pred)
    {
        double sum_squared_errors = 0.0;

        for(int i = 0; i < y_pred.length - 1; i++)
        {
            double error = y_test[i] - y_pred[i];
            sum_squared_errors += pow(error,2);
        }

        return sum_squared_errors/y_test.length;
    }

    public static double rmse(double[] y_test,double[] y_pred)
    {
        double mse = MGDRegressor.mse(y_test,y_pred);

        return Math.sqrt(mse);
    }

    public static double mae(double[] y_test,double[] y_pred)
    {
        double sum_absolute_error = 0.0;
        for(int i = 0; i < y_test.length; i++)
        {
            double error = y_test[i] - y_pred[i];
            sum_absolute_error += Math.abs(error);
        }

        return sum_absolute_error/y_test.length;
    }

    public static double r2_score(double[] y_test, double[] y_pred)
    {
        double sum_y_test = 0;
        for(int i = 0; i< y_test.length; i++)
        {
            sum_y_test += y_test[i];
        }

        double total_sum_squares = getTotalSumSquares(y_test, y_pred, sum_y_test);

        double sum_squared_residuals = 0.0;
        for (int i = 0; i<y_test.length;i++)
        {
            sum_squared_residuals += Math.pow(y_test[i] - y_pred[i],2);
        }

        double r2 = 1 - (sum_squared_residuals/total_sum_squares);

        return r2;
    }

    private static double getTotalSumSquares(double[] y_test, double[] y_pred, double sum_y_test) {
        double mean_y_test = sum_y_test / y_test.length;
        double total_sum_squares = 0.0;

        for(double y_i : y_test)
        {
            total_sum_squares += Math.pow(y_i - mean_y_test,2);
        }

        if(total_sum_squares == 0)
        {
            for (int i = 0; i< y_test.length; i++)
            {
                if(y_test[i] == y_pred[i])
                {
                    total_sum_squares =  1.0;
                }
                total_sum_squares =  0.0;
            }
        }
        return total_sum_squares;
    }

    public static double adjusted_r2(double[] y_test, double[] y_pred, int num_features)
    {
        double r2 = MGDRegressor.r2_score(y_test,y_pred);
        int n = y_test.length;

        double adj_r2 = (1-r2)*(n-1)/(n - num_features - 1);
        if(Double.isNaN(adj_r2))
        {
            return Double.MAX_VALUE;
        }
        return adj_r2;
    }

    public String getParams()
    {
        return "LR=" + this.learning_rate + ", Epochs=" + this.epochs;
    }

    public int getEpochs()
    {
        return epochs;
    }
}

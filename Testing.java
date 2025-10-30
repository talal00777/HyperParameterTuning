package javaml;

import java.util.List;

import static javaml.MGDRegressor.mae;
import static javaml.MGDRegressor.rmse;
import static javaml.MakeRegression.makeRegression;

public class Testing
{
    public static void main(String[] args)
    {
        RegressionData data = makeRegression(100, 3, 20, 13);
        DataSplit ds = new DataSplit(0.2,2);
        List<Object> splits = ds.train_test_split(data.X, data.y);

        double[][] X_train = (double[][]) splits.get(0);
        double[][] X_test = (double[][]) splits.get(1);
        double[] y_train = (double[]) splits.get(2);
        double[] y_test = (double[]) splits.get(3);

        MGDRegressor mgd = new MGDRegressor(200,0.001);
        mgd.train(X_train,y_train);
        double[] y_predictions = mgd.predict(X_test);

        for(int i = 0; i < 10; i++)
        {
            System.out.printf("Predicted value: %.2f, Actual value: %.2f\n", y_predictions[i],y_test[i] );
        }

        System.out.println(mae(y_test,y_predictions));
    }
}

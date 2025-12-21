package main.java.com.tunerapp.tuner;

import main.java.com.tunerapp.data.CrossValidator;
import java.util.Map;

public interface TunableModel<X_TYPE, Y_TYPE> {

    void setParams(Map<String, Object> params);

    /**
     * UPDATED SIGNATURE:
     * Now takes the full training data and a CrossValidator object.
     */
    double fitAndEvaluate(X_TYPE full_X_train, Y_TYPE full_y_train, CrossValidator cv);
}
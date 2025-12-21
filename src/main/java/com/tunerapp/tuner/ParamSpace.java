/*package main.java.com.tunerapp.tuner;

import java.util.*;

public class ParamSpace
{
    private final Map<String, List<? extends Comparable<?>>> space = new LinkedHashMap<>();

    public <T extends Comparable<T>> void add(String name, List<T> values)
    {
        if (values == null || values.isEmpty())
        {
            throw new IllegalArgumentException("Values list cannot be null or empty.");
        }
        Collections.sort(values);
        this.space.put(name, values);
    }

    public List<String> getParamNames()
    {
        return new ArrayList<>(this.space.keySet());
    }

    public List<?> getValues(String name)
    {
        return this.space.get(name);
    }

    public int getParamCount()
    {
        return this.space.size();
    }
}*/


// File: ParamSpace.java (CORRECTED and FINAL Version)
package main.java.com.tunerapp.tuner;

import java.util.*;

/**
 * A class to hold the hyperparameter search space.
 * This version is more flexible to handle data from JSON parsing.
 */
public class ParamSpace {

    private final Map<String, List<? extends Comparable<?>>> space = new LinkedHashMap<>();

    /**
     * A more flexible add method that accepts a list of any comparable objects.
     * It handles sorting internally.
     *
     * @param name The name of the hyperparameter.
     * @param values A list of values to test.
     */
    public <T extends Comparable<T>> void add(String name, List<T> values) {
        if (values == null || values.isEmpty()) {
            throw new IllegalArgumentException("Values list for parameter '" + name + "' cannot be null or empty.");
        }
        List<T> sortedValues = new ArrayList<>(values);
        Collections.sort(sortedValues);
        this.space.put(name, sortedValues);
    }

    public List<String> getParamNames() {
        return new ArrayList<>(this.space.keySet());
    }

    public List<?> getValues(String name) {
        return this.space.get(name);
    }

    public int getParamCount() {
        return this.space.size();
    }
}

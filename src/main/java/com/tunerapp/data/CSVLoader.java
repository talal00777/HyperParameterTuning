package main.java.com.tunerapp.data;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A utility class to load a dataset from a CSV file.
 * This version is overloaded to support loading from a file path (String)
 * or from a generic Reader (like an InputStreamReader for web uploads).
 */
public class CSVLoader {

    /**
     * A simple container to hold the loaded feature matrix (X), target vector (y),
     * and column headers.
     */
    public static class Dataset {
        public final double[][] X;
        public final Object y; // Can be int[] for classification or double[] for regression
        public final String[] headers;

        public Dataset(double[][] X, Object y, String[] headers) {
            this.X = X;
            this.y = y;
            this.headers = headers;
        }
    }

    /**
     * CONVENIENCE METHOD: Loads data from a file path.
     * This is useful for command-line applications and local testing.
     * @param filePath The full path to the CSV file.
     * @param targetColumnName The exact name of the column to be used as the target (y).
     * @param isClassification True if the target is for classification (int[]), false for regression (double[]).
     * @return A Dataset object.
     * @throws IOException If the file cannot be found or read.
     */
    public static Dataset load(String filePath, String targetColumnName, boolean isClassification) throws IOException {
        // This method simply creates a FileReader and delegates to the core load method.
        // The try-with-resources block ensures the FileReader is closed automatically.
        try (Reader reader = new FileReader(filePath)) {
            return load(reader, targetColumnName, isClassification);
        }
    }

    /**
     * CORE METHOD: Loads data from any given Reader.
     * This is the general-purpose method used by the web agent.
     * @param providedReader The Reader object (e.g., an InputStreamReader) to read data from.
     * @param targetColumnName The exact name of the column to be used as the target (y).
     * @param isClassification True if the target is for classification, false for regression.
     * @return A Dataset object.
     * @throws IOException If there is an error during reading.
     */
    public static Dataset load(Reader providedReader, String targetColumnName, boolean isClassification) throws IOException {

        List<double[]> featureList = new ArrayList<>();
        List<Double> targetList = new ArrayList<>();
        String[] headers = null;
        int targetColumnIndex = -1;

        // The CSVParser will be closed automatically by the try-with-resources block.
        try (CSVParser csvParser = new CSVParser(providedReader, CSVFormat.DEFAULT.withFirstRecordAsHeader().withIgnoreEmptyLines(true))) {

            Map<String, Integer> headerMap = csvParser.getHeaderMap();
            if (!headerMap.containsKey(targetColumnName)) {
                throw new IllegalArgumentException("Target column '" + targetColumnName + "' not found in CSV header.");
            }
            targetColumnIndex = headerMap.get(targetColumnName);
            headers = headerMap.keySet().toArray(new String[0]);

            int numFeatures = headers.length - 1;
            if (numFeatures < 1) {
                throw new IllegalArgumentException("CSV must contain at least one feature column besides the target column.");
            }

            for (CSVRecord csvRecord : csvParser) {
                double[] features = new double[numFeatures];
                int featureIdx = 0;
                for (int i = 0; i < headers.length; i++) {
                    String rawValue = csvRecord.get(i);
                    // Robustness: Treat empty or non-parsable values as 0.0 to prevent crashes.
                    double value = 0.0;
                    try {
                        if (rawValue != null && !rawValue.trim().isEmpty()) {
                            value = Double.parseDouble(rawValue);
                        }
                    } catch (NumberFormatException e) {
                        System.err.println("Warning: Could not parse value '" + rawValue + "'. Treating as 0.0.");
                    }

                    if (i == targetColumnIndex) {
                        targetList.add(value);
                    } else {
                        features[featureIdx++] = value;
                    }
                }
                featureList.add(features);
            }
        }

        double[][] X = featureList.toArray(new double[0][]);
        Object y;

        if (isClassification) {
            y = targetList.stream().mapToInt(Double::intValue).toArray();
        } else {
            y = targetList.stream().mapToDouble(Double::doubleValue).toArray();
        }

        return new Dataset(X, y, headers);
    }
}
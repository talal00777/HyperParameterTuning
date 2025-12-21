package main.java.com.tunerapp.mlmodels;

import java.util.*;
import java.util.stream.Collectors;

public final class RandomForestSuite
{

    private RandomForestSuite() {}

    public static class RandomForest
    {

        private final int nEstimators;
        private final int maxDepth;
        private final int minSamplesSplit;
        private final int maxFeatures;
        private List<DecisionTree> forest = new ArrayList<>();
        private final Random random = new Random();

        public RandomForest(int nEstimators, int maxDepth, int minSamplesSplit, int maxFeatures)
        {
            this.nEstimators = nEstimators;
            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
            this.maxFeatures = maxFeatures;
        }

        public void fit(double[][] X, int[] y)
        {
            forest.clear();
            for (int i = 0; i < nEstimators; i++)
            {
                BootstrapSample sample = createBootstrapSample(X, y);
                DecisionTree tree = new DecisionTree(maxDepth, minSamplesSplit, maxFeatures);
                tree.fit(sample.X, sample.y);
                forest.add(tree);
            }
        }

        public int[] predict(double[][] X)
        {
            int[] predictions = new int[X.length];
            for (int i = 0; i < X.length; i++)
            {
                predictions[i] = predictSingle(X[i]);
            }
            return predictions;
        }

        private int predictSingle(double[] x)
        {
            return forest.stream()
                    .mapToInt(tree -> tree.predict(x))
                    .boxed()
                    .collect(Collectors.groupingBy(vote -> vote, Collectors.counting()))
                    .entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .map(Map.Entry::getKey)
                    .orElse(-1);
        }

        private BootstrapSample createBootstrapSample(double[][] X, int[] y)
        {
            int nSamples = X.length;
            int[] indices = random.ints(nSamples, 0, nSamples).toArray();

            double[][] X_sample = new double[nSamples][];
            int[] y_sample = new int[nSamples];

            for (int i = 0; i < nSamples; i++)
            {
                X_sample[i] = X[indices[i]];
                y_sample[i] = y[indices[i]];
            }
            return new BootstrapSample(X_sample, y_sample);
        }

        private static class BootstrapSample
        {
            final double[][] X;
            final int[] y;

            BootstrapSample(double[][] X, int[] y) {
                this.X = X;
                this.y = y;
            }
        }
    }

    private static class DecisionTree
    {

        private Node root;
        private final int maxDepth;
        private final int minSamplesSplit;
        private final int maxFeatures;
        private final Random random = new Random();

        DecisionTree(int maxDepth, int minSamplesSplit, int maxFeatures)
        {
            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
            this.maxFeatures = maxFeatures;
        }

        void fit(double[][] X, int[] y)
        {
            this.root = buildTree(X, y, 0);
        }

        int predict(double[] x)
        {
            return traverseTree(x, this.root);
        }

        private int traverseTree(double[] x, Node node)
        {
            if (node.isLeaf) return node.value;
            return x[node.featureIndex] < node.threshold ? traverseTree(x, node.left) : traverseTree(x, node.right);
        }

        private Node buildTree(double[][] X, int[] y, int depth)
        {
            int nSamples = X.length;
            if (nSamples == 0) return new Node(-1);

            int nFeatures = X[0].length;
            long nClasses = Arrays.stream(y).distinct().count();

            if (depth >= maxDepth || nSamples < minSamplesSplit || nClasses == 1)
            {
                return new Node(mostCommonLabel(y));
            }

            List<Integer> featureIndices = getRandomFeatureIndices(nFeatures);
            Split bestSplit = findBestSplit(X, y, featureIndices);

            if (bestSplit.infoGain <= 0)
            {
                return new Node(mostCommonLabel(y));
            }

            Node leftSubtree = buildTree(bestSplit.X_left, bestSplit.y_left, depth + 1);
            Node rightSubtree = buildTree(bestSplit.X_right, bestSplit.y_right, depth + 1);

            return new Node(bestSplit.featureIndex, bestSplit.threshold, leftSubtree, rightSubtree);
        }

        private List<Integer> getRandomFeatureIndices(int nFeatures)
        {
            List<Integer> allIndices = new ArrayList<>();
            for (int i = 0; i < nFeatures; i++) allIndices.add(i);
            Collections.shuffle(allIndices, random);
            int numFeaturesToSelect = Math.min(this.maxFeatures, nFeatures);
            return allIndices.subList(0, numFeaturesToSelect);
        }

        private Split findBestSplit(double[][] X, int[] y, List<Integer> featureIndices)
        {
            Split bestSplit = new Split(-1);
            double currentGini = calculateGini(y);
            int nSamples = y.length;

            for (int featureIndex : featureIndices)
            {
                Set<Double> uniqueValues = new HashSet<>();
                for (double[] row : X) uniqueValues.add(row[featureIndex]);

                for (double threshold : uniqueValues)
                {
                    List<Integer> leftIndices = new ArrayList<>();
                    List<Integer> rightIndices = new ArrayList<>();
                    for (int i = 0; i < nSamples; i++)
                    {
                        if (X[i][featureIndex] < threshold) leftIndices.add(i);
                        else rightIndices.add(i);
                    }

                    if (leftIndices.isEmpty() || rightIndices.isEmpty()) continue;

                    int[] y_left = getSubset(y, leftIndices);
                    int[] y_right = getSubset(y, rightIndices);

                    double p_left = (double) y_left.length / nSamples;
                    double p_right = (double) y_right.length / nSamples;
                    double weightedGini = p_left * calculateGini(y_left) + p_right * calculateGini(y_right);
                    double infoGain = currentGini - weightedGini;

                    if (infoGain > bestSplit.infoGain)
                    {
                        bestSplit.infoGain = infoGain;
                        bestSplit.featureIndex = featureIndex;
                        bestSplit.threshold = threshold;
                        bestSplit.X_left = getSubset(X, leftIndices);
                        bestSplit.y_left = y_left;
                        bestSplit.X_right = getSubset(X, rightIndices);
                        bestSplit.y_right = y_right;
                    }
                }
            }
            return bestSplit;
        }

        private double calculateGini(int[] y)
        {
            if (y.length == 0) return 0.0;
            Map<Integer, Long> counts = Arrays.stream(y).boxed().collect(Collectors.groupingBy(i -> i, Collectors.counting()));
            double impurity = 1.0;
            for (long count : counts.values())
            {
                impurity -= Math.pow((double) count / y.length, 2);
            }
            return impurity;
        }

        private int mostCommonLabel(int[] y)
        {
            return Arrays.stream(y).boxed()
                    .collect(Collectors.groupingBy(i -> i, Collectors.counting()))
                    .entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .map(Map.Entry::getKey)
                    .orElse(-1); // Return -1 for empty/invalid splits
        }

        private static int[] getSubset(int[] array, List<Integer> indices)
        {
            return indices.stream().mapToInt(i -> array[i]).toArray();
        }

        private static double[][] getSubset(double[][] matrix, List<Integer> indices)
        {
            return indices.stream().map(i -> matrix[i]).toArray(double[][]::new);
        }

        private static class Node
        {
            final int featureIndex;
            final double threshold;
            final Node left;
            final Node right;
            final int value;
            final boolean isLeaf;

            Node(int featureIndex, double threshold, Node left, Node right)
            {
                this.featureIndex = featureIndex; this.threshold = threshold;
                this.left = left; this.right = right;
                this.isLeaf = false; this.value = -1;
            }

            Node(int value)
            {
                this.value = value; this.isLeaf = true;
                this.featureIndex = -1; this.threshold = -1;
                this.left = null; this.right = null;
            }
        }

        private static class Split
        {
            int featureIndex; double threshold; double infoGain;
            double[][] X_left, X_right; int[] y_left, y_right;
            Split(double initialGain) { this.infoGain = initialGain; }
        }
    }

    public static final class ModelMetrics
    {

        private ModelMetrics() {}

        public static double accuracy(int[] y_true, int[] y_pred)
        {
            if (y_true.length != y_pred.length || y_true.length == 0)
            {
                throw new IllegalArgumentException("Input arrays must have the same, non-zero length.");
            }
            long correct = 0;
            for (int i = 0; i < y_true.length; i++)
            {
                if (y_true[i] == y_pred[i]) correct++;
            }
            return (double) correct / y_true.length;
        }

        public static double precision(int[] y_true, int[] y_pred)
        {
            Map<String, Integer> cm = getConfusionMatrixCounts(y_true, y_pred);
            int tp = cm.get("TP");
            int fp = cm.get("FP");
            return (tp + fp) == 0 ? 0.0 : (double) tp / (tp + fp);
        }

        public static double recall(int[] y_true, int[] y_pred)
        {
            Map<String, Integer> cm = getConfusionMatrixCounts(y_true, y_pred);
            int tp = cm.get("TP");
            int fn = cm.get("FN");
            return (tp + fn) == 0 ? 0.0 : (double) tp / (tp + fn);
        }

        public static double f1_score(int[] y_true, int[] y_pred)
        {
            double prec = precision(y_true, y_pred);
            double rec = recall(y_true, y_pred);
            return (prec + rec) == 0 ? 0.0 : 2 * (prec * rec) / (prec + rec);
        }

        private static Map<String, Integer> getConfusionMatrixCounts(int[] y_true, int[] y_pred)
        {
            if (y_true.length != y_pred.length)
            {
                throw new IllegalArgumentException("Input arrays must have same length.");
            }
            Map<String, Integer> counts = new HashMap<>();
            counts.put("TP", 0); counts.put("FP", 0); counts.put("TN", 0); counts.put("FN", 0);

            for (int i = 0; i < y_true.length; i++)
            {
                if (y_true[i] == 1 && y_pred[i] == 1)
                {
                    counts.merge("TP", 1, Integer::sum);
                }
                else if (y_true[i] == 0 && y_pred[i] == 1)
                {
                    counts.merge("FP", 1, Integer::sum);
                }
                else if (y_true[i] == 0 && y_pred[i] == 0)
                {
                    counts.merge("TN", 1, Integer::sum);
                }
                else if (y_true[i] == 1 && y_pred[i] == 0)
                {
                    counts.merge("FN", 1, Integer::sum);
                }
            }
            return counts;
        }
    }
}

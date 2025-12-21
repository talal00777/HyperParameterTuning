package javaml;

class JobResult {
    private final double score; // Now represents MSE (lower is better)
    private final String params;
    private final int epochs;
    public JobResult(double score, String params, int epochs) { this.score = score; this.params = params; this.epochs = epochs; }

    public double getScore() { return score; }
    public String getParams() { return params; }
    public int getEpochs() { return epochs;}
}

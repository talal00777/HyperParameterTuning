package javaml;

class JobResult {
    private final double score; // Now represents MSE (lower is better)
    private final String params;
    public JobResult(double score, String params) { this.score = score; this.params = params; }
    public double getScore() { return score; }
    public String getParams() { return params; }
}


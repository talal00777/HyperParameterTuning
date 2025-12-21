package main.java.com.tunerapp.agent;

import java.util.Map;

// This is just a simple class to hold the state of a job
public class JobStatus {
    public String jobId;
    public String status; // e.g., "QUEUED", "RUNNING", "COMPLETED", "FAILED"
    public String message;
    public Map<String, Object> bestParams;
    public Double bestScore;

    public JobStatus(String jobId, String status, String message) {
        this.jobId = jobId;
        this.status = status;
        this.message = message;
    }
    // Getters and setters would be good practice here, but public fields are fine for now.
}